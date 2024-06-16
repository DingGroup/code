#!/home/xqding/apps/miniconda3/envs/jop/bin/python

#SBATCH --job-name=uq
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-24
#SBATCH --output=./slurm_output/uq_%a.out
#SBATCH --open-mode=truncate

import mdtraj
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import sys
import os
import pickle
import pandas as pd
sys.path.append('../HPS_Urry/script')
from functions import *

params = pd.read_csv(
    "../HPS_Urry/data/params_hps_Urry.csv", header=0, index_col=0,
)

with open(f"../HPS_Urry_PC_2/output/params/single_type_params_optimized.pkl", "rb") as f:
    lamb = pickle.load(f)

params['lambda'] = lamb

with open("../HPS_Urry/data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

data = pd.read_table('../HPS_Urry/data/Rg.txt', header=0, sep='\s+')
idx_protein = int(os.environ['SLURM_ARRAY_TASK_ID'])

protein_name = data.loc[idx_protein, 'name']
seq = seq[protein_name]
kappa = data.loc[idx_protein, 'kappa_nm_inverse']

top = mdtraj.load_psf(f'../HPS_Urry/output/psf/{protein_name}.psf')

traj_data = mdtraj.load_dcd(f'./output/traj_data_and_noise/{protein_name}_data.dcd', top=top)
traj_noise = mdtraj.load_dcd(f'./output/traj_data_and_noise/{protein_name}_noise.dcd', top=top)

system = make_system(seq, params, kappa)
integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
platform = mm.Platform.getPlatformByName('Reference')
context = mm.Context(system, integrator, platform)

## compute reduced potential energy of traj_data and traj_noise
kbT = 300*unit.kelvin*unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
def compute_u(traj):
    u = []
    for idx_frame in range(traj.n_frames):
        context.setPositions(traj.xyz[idx_frame])
        u.append(context.getState(getEnergy=True).getPotentialEnergy()/kbT)
    return np.array(u)

print('compute u on data', flush=True)
u_data = compute_u(traj_data)

print('compute u on noise', flush=True)
u_noise = compute_u(traj_noise)

os.makedirs('./output/uq', exist_ok=True)
with open(f'./output/uq/{protein_name}.pkl', 'wb') as f:
    pickle.dump({'data': u_data, 'noise': u_noise}, f)