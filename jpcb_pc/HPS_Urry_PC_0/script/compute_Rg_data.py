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
    traj = mdtraj.load_dcd(f'./output/traj_data_and_noise/{protein_name}_data.dcd', top=top)
    rg = mdtraj.compute_rg(traj, masses=mass)

    os.makedirs('./output/Rg', exist_ok=True)
    with open(f'./output/Rg/{protein_name}_data.pkl', 'wb') as f:
        pickle.dump(rg, f)

    print(f'{protein_name} is done')        
    