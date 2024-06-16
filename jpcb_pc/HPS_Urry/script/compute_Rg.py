import mdtraj
import numpy as np
import sys
import pickle
import pandas as pd
from dltoolbox.aa_code import aa_code_123, aa_code_321, aa_mass
import matplotlib.pyplot as plt
import os
from sys import exit

with open("./data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

data = pd.read_table('./data/Rg.txt', header=0, sep='\s+')

rg_sim = []
for idx_protein in range(data.shape[0]):
    protein_name = data.loc[idx_protein, 'name']
    mass = np.array([aa_mass[aa] for aa in seq[protein_name]])

    top = mdtraj.load_psf(f'./output/psf/{protein_name}.psf')
    T = 300
    traj = mdtraj.load_dcd(f'./output/traj/{protein_name}_{T:.2f}.dcd', top=top)
    traj = traj[10000::10]

    rg = mdtraj.compute_rg(traj, mass)
    print(f"{protein_name}, {rg.mean():.3f}, {rg.min():.3f}, {rg.max():.3f}, {data.loc[idx_protein, 'Rg_nm']:.3f}")
    rg_sim.append(rg.mean())

with open('./output/fig/rg_sim.pkl', 'wb') as f:
    pickle.dump(rg_sim, f)

fig = plt.figure()
plt.plot(data['Rg_nm'], rg_sim, 'o')
plt.xlim(1, 4.6)
plt.ylim(1, 4.6)
plt.xlabel('Rg_exp (nm)')
plt.ylabel('Rg_sim (nm)')
ax = plt.gca()
ax.set_aspect(1)
plt.plot([0, 5], [0, 5], '--')
plt.tight_layout()
plt.savefig(f'./output/fig/Rg_sim.png')