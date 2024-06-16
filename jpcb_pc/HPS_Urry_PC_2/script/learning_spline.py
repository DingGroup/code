import numpy as np
import pandas as pd
import pickle
import argparse
import os
from functools import partial
from sys import exit
from scipy import optimize
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding


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
    with open(f"./output/spline_basis/{protein_name}.pkl", "rb") as file_handle:
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
    if not os.path.isfile(f"./output/spline_basis/{names[idx_protein]}.pkl"):
        continue
    protein_name = names[idx_protein]
    data_list.append(get_data(protein_name))

    print(f"protein {protein_name} is done", flush=True)

for i in range(len(data_list)):
    basis = np.concatenate((data_list[i]['basis_data'], data_list[i]['basis_noise']), axis = 0)
    data_list[i]['basis'] = basis.reshape((basis.shape[0], -1))
    del data_list[i]['basis_data']
    del data_list[i]['basis_noise']

    data_list[i]['index'] = np.ones(data_list[i]['basis'].shape[0], dtype = np.int64) * i

    intercept = np.concatenate((data_list[i]['intercept_data'], data_list[i]['intercept_noise']), axis = 0)
    data_list[i]['intercept'] = intercept
    del data_list[i]['intercept_data']
    del data_list[i]['intercept_noise']

    uq = np.concatenate((data_list[i]['uq_data'], data_list[i]['uq_noise']), axis = 0)
    data_list[i]['uq'] = uq
    del data_list[i]['uq_data']
    del data_list[i]['uq_noise']

    data_list[i]['y'] = np.concatenate([np.ones(data_list[i]['basis'].shape[0] // 2), np.zeros(data_list[i]['basis'].shape[0] // 2)])

basis = np.concatenate([data['basis'] for data in data_list], axis = 0)
intercept = np.concatenate([data['intercept'] for data in data_list], axis = 0)
uq = np.concatenate([data['uq'] for data in data_list], axis = 0)
y = np.concatenate([data['y'] for data in data_list], axis = 0)
index = np.concatenate([data['index'] for data in data_list], axis = 0)

@jax.jit
@partial(jax.value_and_grad, argnums=[0,1])
def compute_loss(theta, F, basis, intercept, uq, y, index):
    up = jnp.matmul(basis, theta) + intercept - F[index]
    loss = optax.sigmoid_binary_cross_entropy(-(up - uq), y).mean()
    return loss

with open('./output/spline_basis/init_theta.pkl', 'rb') as file_handle:
    init_theta = pickle.load(file_handle)

theta = init_theta.reshape(-1)
F = np.zeros(len(data_list))

sharding = PositionalSharding(jax.devices()).reshape(4, 1)
basis = jax.device_put(basis, sharding)
intercept = jax.device_put(intercept, sharding.reshape(-1))
uq = jax.device_put(uq, sharding.reshape(-1))
y = jax.device_put(y, sharding.reshape(-1))
index = jax.device_put(index, sharding.reshape(-1))

# theta = jax.device_put(theta, sharding.replicate())
# F = jax.device_put(F, sharding.replicate())

# exit()

x_init = np.concatenate([theta, F])

def compute_loss_and_grad(x):
    theta = x[0:-len(data_list)]
    F = x[-len(data_list):]

    theta = jax.device_put(theta, sharding.replicate())
    F = jax.device_put(F, sharding.replicate())

    loss, (grad_theta, grad_F) = compute_loss(theta, F, basis, intercept, uq, y, index)

    loss = np.array(loss)
    grad = np.concatenate([np.array(grad_theta), np.array(grad_F)])
    return loss, grad

options = {"disp": True, "gtol": 1e-12}
results = optimize.minimize(
    compute_loss_and_grad,
    x_init,
    jac=True,
    method="L-BFGS-B",
    tol=1e-20,
    options=options,
)

with open(f'./output/params/spline_params_optimized.pkl', 'wb') as f:
    pickle.dump(results.x[0:-len(data_list)].reshape((210, 16)), f)


exit()

@jax.jit
@partial(jax.value_and_grad, argnums=[0,1])
def compute_loss_per_protein(theta, F, data):
    basis_data, basis_noise = data["basis_data"], data["basis_noise"]
    intercept_data, intercept_noise = data["intercept_data"], data["intercept_noise"]

    basis_data = basis_data.reshape((basis_data.shape[0], -1))
    basis_noise = basis_noise.reshape((basis_noise.shape[0], -1))

    up_data = jnp.matmul(basis_data, theta) + intercept_data
    up_noise = jnp.matmul(basis_noise, theta) + intercept_noise
    up = jnp.concatenate([up_data, up_noise])
    up = up - F

    uq = jnp.concatenate([data["uq_data"], data["uq_noise"]])

    y = jnp.concatenate([jnp.ones(up_data.shape[0]), jnp.zeros(up_noise.shape[0])])

    loss = optax.sigmoid_binary_cross_entropy(-(up - uq), y).mean()

    return loss

def compute_loss_and_grad(x):
    theta = x[0:-len(data_list)]
    Fs = x[-len(data_list):]
    loss_list = []
    grad_theta_list = []
    grad_F_list = []
    for i in range(len(data_list)):
        loss, (grad_theta, grad_F) = jax.jit(compute_loss_per_protein)(theta, Fs[i], jax.device_put(data_list[i]))
        loss_list.append(loss)
        grad_theta_list.append(grad_theta)
        grad_F_list.append(grad_F)

    loss = jnp.stack(loss_list).mean(axis=0)
    grad_theta = jnp.stack(grad_theta_list).mean(axis=0)
    grad_F = jnp.stack(grad_F_list)
    grad = jnp.concatenate([grad_theta, grad_F])
    return loss, grad

#f = jax.jit(jax.value_and_grad(compute_loss, argnums=0))
#f = jax.value_and_grad(compute_loss, argnums=0)

with open('./output/spline_basis/init_theta.pkl', 'rb') as file_handle:
    init_theta = pickle.load(file_handle)

theta = init_theta.reshape(-1)
x_init = np.concatenate([theta, np.zeros(len(data_list))])

options = {"disp": True, "gtol": 1e-12}
results = optimize.minimize(
    lambda x: [np.array(r) for r in compute_loss_and_grad(jnp.array(x))],
    x_init,
    jac=True,
    method="L-BFGS-B",
    tol=1e-20,
    options=options,
)

exit()

lamb_optimized = results.x[0:-len(data_list)]

os.makedirs("./output/params", exist_ok=True)
with open(f"./output/params/{args.type}_type_params_optimized.pkl", "wb") as f:
    pickle.dump(lamb_optimized, f)

loss = compute_loss(x)

bounds = []
for i in range(210):
    for j in range(16):
        if j == 0:
            bounds.append((5, None))
        else:
            bounds.append((None, None))
bounds += [(None, None)] * len(data_list)