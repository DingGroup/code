#!/home/xqding/apps/miniconda3/envs/jop/bin/python

#SBATCH --job-name=md
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:gtx1080:1
#SBATCH --array=0-5
#SBATCH --mem-per-cpu=1000M
#SBATCH --output=./slurm_output/md_%a.out

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
import argparse
import sys
import pandas as pd
import os
import time

params = pd.read_csv(
    "../HPS_Urry/data/params_hps_Urry.csv", header=0, index_col=0,
)

## protein sequences
with open("../HPS_Urry/data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

data = pd.read_table('../HPS_Urry/data/Rg.txt', header=0, sep='\s+')


with open("./output/params/double_type_params_optimized.pkl", "rb") as f:
    lamb = pickle.load(f)

idx_protein = 1

protein_name = data.loc[idx_protein, 'name']
seq = seq[protein_name]
kappa = data.loc[idx_protein, 'kappa_nm_inverse']

Ts = np.exp(np.linspace(np.log(800), np.log(300), 6))
#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
task_id = -1
T = Ts[task_id]
print("T = ", T)

system = make_system(seq, params, kappa)
#system.removeForce(2)

k = 0
for i in range(params.shape[0]):
    for j in range(i, params.shape[0]):
        lamb[k] = 0.5*(params.iloc[i, -1] + params.iloc[j, -1])
        k += 1
hps = make_hps_double_type(seq, params, lamb)
#system.addForce(hps)


# ## serizlize system using Openmm
# os.makedirs("./output/system", exist_ok=True)
# with open(f"./output/system/{protein_name}.xml", "w") as f:
#     f.write(mm.XmlSerializer.serialize(system))

topology = make_topology(seq)
integrator = mm.LangevinMiddleIntegrator(
    T * unit.kelvin,
    1.0 / unit.picoseconds,
    10.0 * unit.femtoseconds,
)
platform_name = "Reference"
platform = mm.Platform.getPlatformByName(platform_name)
initial_position = make_initial_position(seq)
simulation = app.Simulation(topology, system, integrator, platform)
simulation.context.setPositions(initial_position)

## get potential energy
print('initial energy')
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

exit()

simulation.minimizeEnergy()

print('initial equilibration')
simulation.step(10_000_000)

simulation.reporters.append(
    app.DCDReporter(f"./output/traj/{protein_name}_{T:.2f}_regular.dcd", 1000)
)

print('Start simulation')
start_time = time.time()
simulation.step(20_000_000)
end_time = time.time()
print(f"Time elapsed: {end_time - start_time:.2f} seconds")

exit()
