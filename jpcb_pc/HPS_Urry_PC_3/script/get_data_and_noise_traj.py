import mdtraj
import pandas as pd
import numpy as np
from dltoolbox.aa_code import aa_mass
import pickle
import os

with open("../HPS_Urry/data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

data = pd.read_table('../HPS_Urry/data/Rg.txt', header=0, sep='\s+')

for idx_protein in range(data.shape[0]):
    protein_name = data.loc[idx_protein, 'name']
    mass = np.array([aa_mass[aa] for aa in seq[protein_name]])
    top = mdtraj.load_psf(f'../HPS_Urry/output/psf/{protein_name}.psf')
    T = 300
    traj = mdtraj.load_dcd(f'../HPS_Urry_PC_2/output/traj_single_type/{protein_name}_{T:.2f}.dcd', top=top)
    traj = traj[10000:]

    N = len(traj)
    data_traj = traj[0:N//2:5]
    noise_traj = traj[N//2::5]

    os.makedirs('./output/traj_data_and_noise', exist_ok=True)
    data_traj.save(f'./output/traj_data_and_noise/{protein_name}_data.dcd')
    noise_traj.save(f'./output/traj_data_and_noise/{protein_name}_noise.dcd')

    print(f'{protein_name} is done')