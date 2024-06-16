import pandas as pd
import openmm as mm
import openmm.unit as unit
import openmm.app as app
import numpy as np
from sys import exit
import pickle
from functions import *
import ray
import argparse
import sys
from mmray import *
import pandas as pd
import os
import time
import pickle

## force field parameters
params = pd.read_csv(
    "./data/params_hps_Urry.csv",
    header=0,
    index_col=0,
)

## protein sequences
with open("./data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

## protein name, epxerimental Rg and kappa
data = pd.read_table("./data/Rg.txt", header=0, sep="\s+")

## protein index from command line
parser = argparse.ArgumentParser()
parser.add_argument("--idx_protein", type=int)
args = parser.parse_args()
idx_protein = args.idx_protein

protein_name = data.loc[idx_protein, "name"]
seq = seq[protein_name]
kappa = data.loc[idx_protein, "kappa_nm_inverse"]

Ts = np.exp(np.linspace(np.log(1000), np.log(300), 10))
actors = []

ray.init()
for T in Ts:
    system = make_system(seq, params, kappa)
    topology = make_topology(seq)
    integrator = mm.LangevinMiddleIntegrator(
        T * unit.kelvin,
        1.0 / unit.picoseconds,
        10.0 * unit.femtoseconds,
    )
    platform_name = "CUDA"
    initial_position = make_initial_position(seq)
    os.makedirs(f"./output/traj", exist_ok=True)
    reporters = {
        "DCD": {
            "file": f"./output/traj/{protein_name}_{T:.2f}.dcd",
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

with open(f"./output/traj/{protein_name}_TRE.pkl", "wb") as f:
    pickle.dump({"record": tre.record, "accept_rate": tre.accept_rate}, f)

ray.shutdown()
