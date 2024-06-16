import pandas as pd
import openmm as mm
import openmm.unit as unit
import openmm.app as app
import numpy as np
from sys import exit
import pickle
import sys

sys.path.append("../HPS_Urry/script")
from functions import *

sys.path.append("./script")
sys.path.append("/home/xqding/my_projects_on_github/pccg")
from pccg.utils import spline
from utils import *
import ray
import argparse
import sys
from mmray import *
import pandas as pd
import os
import time
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--idx_protein", type=int)
parser.add_argument("--wc", type=float)
args = parser.parse_args()

## force field parameters
params = pd.read_csv(
    "../HPS_Urry/data/params_hps_Urry.csv",
    header=0,
    index_col=0,
)


with open(f"./output/params/params_optimized_wc_{args.wc:.2E}.pkl", "rb") as f:
    spline_coeff = pickle.load(f)
r = np.linspace(0.0, 2.0, 1000)
basis = spline.bs_nb(r, 2.0, 12)
unb = np.matmul(spline_coeff, basis.T)

k = 0
unb_expand = np.zeros((params.shape[0], params.shape[0], unb.shape[-1]))
for i in range(params.shape[0]):
    for j in range(i, params.shape[0]):
        unb_expand[i, j] = unb[k]
        unb_expand[j, i] = unb[k]
        k += 1
unb = unb_expand

## protein sequences
with open("../HPS_Urry/data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

## protein name, epxerimental Rg and kappa
data = pd.read_table("../HPS_Urry/data/Rg.txt", header=0, sep="\s+")

idx_protein = args.idx_protein

protein_name = data.loc[idx_protein, "name"]
seq = seq[protein_name]
kappa = data.loc[idx_protein, "kappa_nm_inverse"]

Ts = np.exp(np.linspace(np.log(1000), np.log(300), 10))
actors = []

ray.init()
for T in Ts:
    system = make_system(seq, params, kappa)
    system.removeForce(2)
    hps = make_hps_spline(seq, params, unb)
    system.addForce(hps)

    topology = make_topology(seq)
    integrator = mm.LangevinMiddleIntegrator(
        T * unit.kelvin,
        1.0 / unit.picoseconds,
        10.0 * unit.femtoseconds,
    )
    platform_name = "CUDA"
    initial_position = make_initial_position(seq)
    os.makedirs(f"./output/traj_wc_{args.wc:.2E}", exist_ok=True)
    reporters = {
        "DCD": {
            "file": f"./output/traj_wc_{args.wc:.2E}/{protein_name}_{T:.2f}.dcd",
            "reportInterval": 1000,
        },
    }
    actor = TREActor.options(num_cpus=1, num_gpus=1).remote(
        topology, system, integrator, platform_name, initial_position, reporters
    )
    actors.append(actor)

for actor in actors:
    actor.minimize_energy.remote()

tre = TRE(actors)

print("running equilibration")
tre.run(10_000_000, 0)

print("running temperature replica exchange")
start_time = time.time()
tre.run(100_000_000, 1000)
end_time = time.time()

print(f"Time elapsed: {end_time - start_time:.2f} seconds")
print("accept_rate", tre.accept_rate)

with open(f"./output/traj_wc_{args.wc:.2E}/{protein_name}_TRE.pkl", "wb") as f:
    pickle.dump({"record": tre.record, "accept_rate": tre.accept_rate}, f)

ray.shutdown()