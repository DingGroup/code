import numpy as np
import pandas as pd
import pickle
import argparse
import os
from sys import exit
from scipy import optimize
import cvxpy as cp


parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, choices=["single", "double"])

args = parser.parse_args()

params = pd.read_csv(
    "../HPS_Urry/data/params_hps_Urry.csv",
    header=0,
    index_col=0,
)

with open("../HPS_Urry/data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

data = pd.read_table("../HPS_Urry/data/Rg.txt", header=0, sep="\s+")
names = data["name"].values


def get_data(protein_name):
    with open(f"./output/basis/{args.type}_type_{protein_name}.pkl", "rb") as file_handle:
        data = pickle.load(file_handle)
        basis_data = data["basis_data"]
        basis_noise = data["basis_noise"]
        intercept_data = data["intercept_data"]
        intercept_noise = data["intercept_noise"]

    with open(f"./output/uq/{protein_name}.pkl", "rb") as f:
        uq = pickle.load(f)
        uq_data = uq["data"]
        uq_noise = uq["noise"]

    with open(f"./output/ME/{protein_name}_weight.pkl", "rb") as f:
        weight_data = pickle.load(f)

    select_idx = np.random.choice(
        basis_data.shape[0], basis_data.shape[0], replace=True, p=weight_data
    )
    basis_data = basis_data[select_idx]
    intercept_data = intercept_data[select_idx]
    uq_data = uq_data[select_idx]

    return {
        "basis_data": basis_data,
        "basis_noise": basis_noise,
        "intercept_data": intercept_data,
        "intercept_noise": intercept_noise,
        "uq_data": uq_data,
        "uq_noise": uq_noise,
    }


data_list = []
np.random.seed(0)
for idx_protein in range(len(names)):
    protein_name = names[idx_protein]
    data_list.append(get_data(protein_name))


if args.type == "single":
    lamb = np.array(params["lambda"].values)
elif args.type == "double":
    lamb = []
    for i in range(params.shape[0]):
        for j in range(i, params.shape[0]):
            aa_i, aa_j = params.index[i], params.index[j]
            lamb.append((params.loc[aa_i, 'lambda'] + params.loc[aa_j, 'lambda'])/2)
    lamb = np.array(lamb)        

theta = cp.Variable(len(lamb))
F = cp.Variable(len(data_list))

theta.value = lamb
F.value = np.zeros(len(data_list))

loss_total = 0
for i in range(len(data_list)):
    data = data_list[i]
    basis = np.concatenate([data["basis_data"], data["basis_noise"]], axis=0)
    intecept = np.concatenate([data["intercept_data"], data["intercept_noise"]], axis=0)
    up = basis @ theta + intecept

    uq = np.concatenate([data["uq_data"], data["uq_noise"]], axis=0)
    
    logit = up - F[i] - uq
    y = np.concatenate([np.ones(data["basis_data"].shape[0]), np.zeros(data["basis_noise"].shape[0])], axis=0)
    loss = cp.multiply(y, cp.logistic(logit)) + cp.multiply(1-y, cp.logistic(-logit))
    loss_total += cp.mean(loss)
constraints = [theta >= 0, theta <= 1]
prob = cp.Problem(cp.Minimize(loss_total), constraints)
prob.solve(solver = 'CLARABEL',verbose=True)

exit()



F = np.zeros(1)

exit()

def compute_loss_per_protein(lamb, F, data):
    up_data = jnp.matmul(data["basis_data"], lamb) + data["intercept_data"]
    up_noise = jnp.matmul(data["basis_noise"], lamb) + data["intercept_noise"]
    up = jnp.concatenate([up_data, up_noise])
    up = up - F

    uq = jnp.concatenate([data["uq_data"], data["uq_noise"]])

    y = jnp.concatenate([jnp.ones(up_data.shape[0]), jnp.zeros(up_noise.shape[0])])

    loss = optax.sigmoid_binary_cross_entropy(-(up - uq), y).mean()

    return loss


def compute_loss(x):
    lamb = x[0:-len(data_list)]
    Fs = x[-len(data_list):]
    loss = 0
    for i in range(len(data_list)):
        loss += compute_loss_per_protein(lamb, Fs[i], data_list[i])
    return loss


f = jax.jit(jax.value_and_grad(compute_loss, argnums=0))

x_init = jnp.concatenate([lamb, jnp.zeros(len(data_list))])

options = {"disp": True, "gtol": 1e-12}
results = optimize.minimize(
    lambda x: [np.array(r) for r in f(jnp.array(x))],
    x_init,
    jac=True,
    method="L-BFGS-B",
    bounds=[(0, 1.0)] * len(lamb) + [(None, None)] * len(data_list),
    tol=1e-20,
    options=options,
)

lamb_optimized = results.x[0:-len(data_list)]

os.makedirs("./output/params", exist_ok=True)
with open(f"./output/params/{args.type}_type_params_optimized.pkl", "wb") as f:
    pickle.dump(lamb_optimized, f)