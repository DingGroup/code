import pandas as pd
import openmm as mm
import openmm.unit as unit
import openmm.app as app
import numpy as np
from sys import exit
import pickle
import sys
sys.path.append('../HPS_Urry/script')
from functions import *
sys.path.append('./script')
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
parser.add_argument("--type", type=str, choices=["single", "double"])
args = parser.parse_args()

## force field parameters
params = pd.read_csv(
    "../HPS_Urry/data/params_hps_Urry.csv", header=0, index_col=0,
)

with open(f"./output/params/{args.type}_type_params_optimized.pkl", "rb") as f:
    lamb = pickle.load(f)

## protein sequences
with open("../HPS_Urry/data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

## protein name, epxerimental Rg and kappa
data = pd.read_table('../HPS_Urry/data/Rg.txt', header=0, sep='\s+')

idx_protein = args.idx_protein

protein_name = data.loc[idx_protein, "name"]
seq = seq[protein_name]
kappa = data.loc[idx_protein, "kappa_nm_inverse"]

Ts = np.exp(np.linspace(np.log(1000), np.log(300), 10))
actors = []

ray.init()
for T in Ts:
    if args.type == "single":
        params['lambda'] = lamb
        system = make_system(seq, params, kappa)
    elif args.type == "double":
        system = make_system(seq, params, kappa)
        system.removeForce(2)
        hps = make_hps_double_type(seq, params, lamb)
        system.addForce(hps)
    
    topology = make_topology(seq)
    integrator = mm.LangevinMiddleIntegrator(
        T * unit.kelvin,
        1.0 / unit.picoseconds,
        10.0 * unit.femtoseconds,
    )
    platform_name = "CUDA"
    initial_position = make_initial_position(seq)
    os.makedirs(f"./output/traj_{args.type}_type", exist_ok=True)
    reporters = {
        "DCD": {
            "file": f"./output/traj_{args.type}_type/{protein_name}_{T:.2f}.dcd",
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

with open(f"./output/traj_{args.type}_type/{protein_name}_TRE.pkl", "wb") as f:
    pickle.dump({"record": tre.record, "accept_rate": tre.accept_rate}, f)

ray.shutdown()
