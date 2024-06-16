#!/home/xqding/apps/miniconda3/envs/jop/bin/python

#SBATCH --job-name=spline
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-16
#SBATCH --output=./slurm_output/spline_%a.out
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
from dltoolbox.aa_code import aa_code_123, aa_code_321
from collections import defaultdict

sys.path.append("../HPS_Urry/script")
from functions import *
sys.path.append("/home/xqding/my_projects_on_github/pccg")
from pccg.utils import spline

params = pd.read_csv(
    "../HPS_Urry/data/params_hps_Urry.csv",
    header=0,
    index_col=0,
)

with open("../HPS_Urry/data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

data = pd.read_table("../HPS_Urry/data/Rg.txt", header=0, sep="\s+")

idx_protein = int(os.environ['SLURM_ARRAY_TASK_ID'])

protein_name = data.loc[idx_protein, "name"]
seq = seq[protein_name]
kappa = data.loc[idx_protein, "kappa_nm_inverse"]

top = mdtraj.load_psf(f"../HPS_Urry/output/psf/{protein_name}.psf")

traj_data = mdtraj.load_dcd(f"./output/traj_data_and_noise/{protein_name}_data.dcd", top=top)
traj_noise = mdtraj.load_dcd(f"./output/traj_data_and_noise/{protein_name}_noise.dcd", top=top)

kbT = 300 * unit.kelvin * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

def compute_basis_for_one_interaction(i, j, traj):
    aa_i = aa_code_123[seq[i]]
    aa_j = aa_code_123[seq[j]]
    sigma = 0.5 * (params.loc[aa_i, "sigma"] + params.loc[aa_j, "sigma"])

    r = mdtraj.compute_distances(traj, [[i, j]], periodic=False).flatten()
    knots = np.linspace(0, 2.0, 13)
    boundary_knots = knots[[0, -1]]
    knots = knots[1:-1]
    basis = spline.bs(r, knots, boundary_knots)

    basis[np.isnan(basis)] = 0.

    eps = 0.8368
    intercept = 4*eps*((sigma/r)**12)

    basis = basis / kbT.value_in_unit(unit.kilojoule_per_mole)
    intercept = intercept / kbT.value_in_unit(unit.kilojoule_per_mole)

    return basis, intercept

def compute_basis(traj):
    basis = defaultdict(lambda: np.zeros((traj.n_frames, 16)))
    intercept = 0
    for i in range(len(seq)):
        for j in range(i + 2, len(seq)):
            b, t = compute_basis_for_one_interaction(i, j, traj)
            aa_i, aa_j = sorted([seq[i], seq[j]])
            basis[(aa_i, aa_j)] += b
            intercept += t
    return basis, intercept

print("compute basis on data", flush=True)
basis_data, intercept_data = compute_basis(traj_data)

print("compute basis on noise", flush=True)
basis_noise, intercept_noise = compute_basis(traj_noise)

## compute bond and electrostatic energy
system = make_system(seq, params, kappa)
integrator = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
)
platform = mm.Platform.getPlatformByName("Reference")
context = mm.Context(system, integrator, platform)

## compute reduced potential energy of traj_data and traj_noise
def compute_bond_and_elec(traj):
    u = []
    for idx_frame in range(traj.n_frames):
        context.setPositions(traj.xyz[idx_frame])
        u.append(
            context.getState(getEnergy=True, groups={1, 2}).getPotentialEnergy() / kbT
        )
    return np.array(u)

intercept_data += compute_bond_and_elec(traj_data)
intercept_noise += compute_bond_and_elec(traj_noise)

basis_data_list = []
basis_noise_list = []
for i in range(params.shape[0]):
    for j in range(i, params.shape[0]):
        aa_i, aa_j = aa_code_321[params.index[i]], aa_code_321[params.index[j]]
        aa_i, aa_j = sorted([aa_i, aa_j])
        basis_data_list.append(basis_data[(aa_i, aa_j)])
        basis_noise_list.append(basis_noise[(aa_i, aa_j)])

basis_data = np.stack(basis_data_list, axis=1)
basis_noise = np.stack(basis_noise_list, axis=1)

os.makedirs("./output/spline_basis", exist_ok=True)
with open(f"./output/spline_basis/{protein_name}.pkl", "wb") as file_handle:
    pickle.dump({
        'basis_data': basis_data,
        'basis_noise': basis_noise,
        'intercept_data': intercept_data,
        'intercept_noise': intercept_noise,
    }, file_handle)

exit()

with open(f"./output/uq/{protein_name}.pkl", "rb") as file_handle:
    uq = pickle.load(file_handle)
    uq_data = uq["data"]
    uq_noise = uq["noise"]

with open('./output/spline_basis/init_theta.pkl', 'rb') as file_handle:
    init_theta = pickle.load(file_handle)
        