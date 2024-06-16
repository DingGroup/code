#!/home/xqding/apps/miniconda3/envs/jop/bin/python

#SBATCH --job-name=basis
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-24
#SBATCH --output=./slurm_output/basis_%a.out
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

params = pd.read_csv(
    "../HPS_Urry/data/params_hps_Urry.csv",
    header=0,
    index_col=0,
)

with open(f"../HPS_Urry_PC_1/output/params/single_type_params_optimized.pkl", "rb") as f:
    lamb = pickle.load(f)

params['lambda'] = lamb

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
    epsilon = 0.8368

    r = mdtraj.compute_distances(traj, [[i, j]], periodic=False).flatten()
    r = r.astype(np.float64)
    ulj = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    r0 = 2 ** (1.0 / 6) * sigma
    basis = np.heaviside(r - r0, 0) * ulj - np.heaviside(r0 - r, 0) * epsilon
    intercept = np.heaviside(r0 - r, 0) * (ulj + epsilon)

    r_cut = 4.0
    basis = np.heaviside(r_cut - r, 0) * basis
    intercept = np.heaviside(r_cut - r, 0) * intercept

    basis = basis / kbT.value_in_unit(unit.kilojoule_per_mole)
    intercept = intercept / kbT.value_in_unit(unit.kilojoule_per_mole)

    return basis, intercept


def compute_basis(traj):
    basis = defaultdict(lambda: np.zeros(traj.n_frames))
    intercept = 0
    for i in range(len(seq)):
        for j in range(i + 2, len(seq)):
            b, t = compute_basis_for_one_interaction(i, j, traj)

            aa_i, aa_j = sorted([seq[i], seq[j]])
            basis[aa_i] += b/2.
            basis[aa_j] += b/2.
                        
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

single_type_basis_data = np.stack([basis_data[aa_code_321[n]] for n in params.index], axis=1)
single_type_basis_noise = np.stack([basis_noise[aa_code_321[n]] for n in params.index], axis=1)

signle_type_lamb = params["lambda"].values
up_data = np.matmul(single_type_basis_data, signle_type_lamb) + intercept_data
up_noise = np.matmul(single_type_basis_noise, signle_type_lamb) + intercept_noise

with open(f"./output/uq/{protein_name}.pkl", "rb") as file_handle:
    uq = pickle.load(file_handle)
    uq_data = uq["data"]
    uq_noise = uq["noise"]

assert np.allclose(up_data, uq_data, atol = 1e-3)
assert np.allclose(up_noise, uq_noise, atol = 1e-3)

os.makedirs("./output/basis", exist_ok=True)
with open(f"./output/basis/single_type_{protein_name}.pkl", "wb") as file_handle:
    pickle.dump(
        {
            "basis_data": single_type_basis_data,
            "basis_noise": single_type_basis_noise,
            "intercept_data": intercept_data,
            "intercept_noise": intercept_noise,
        },
        file_handle,
    )

double_type_basis_data = []
double_type_basis_noise = []
double_type_lamb = []
for i in range(params.shape[0]):
    for j in range(i, params.shape[0]):
        aa_i, aa_j = sorted([params.index[i], params.index[j]])
        double_type_lamb.append((params.loc[aa_i, "lambda"] + params.loc[aa_j, "lambda"]) / 2.)

        aa_i, aa_j = sorted([aa_code_321[aa_i], aa_code_321[aa_j]])
        double_type_basis_data.append(basis_data[(aa_i, aa_j)])
        double_type_basis_noise.append(basis_noise[(aa_i, aa_j)])        

double_type_basis_data = np.stack(double_type_basis_data, axis=1)
double_type_basis_noise = np.stack(double_type_basis_noise, axis=1)
double_type_lamb = np.array(double_type_lamb)

up_data = np.matmul(double_type_basis_data, double_type_lamb) + intercept_data
up_noise = np.matmul(double_type_basis_noise, double_type_lamb) + intercept_noise

assert np.allclose(up_data, uq_data, atol = 1e-3)
assert np.allclose(up_noise, uq_noise, atol = 1e-3)

with open(f"./output/basis/double_type_{protein_name}.pkl", "wb") as file_handle:
    pickle.dump(
        {
            "basis_data": double_type_basis_data,
            "basis_noise": double_type_basis_noise,
            "intercept_data": intercept_data,
            "intercept_noise": intercept_noise,
        },
        file_handle,
    )