import mdtraj
import pandas as pd
import numpy as np
from dltoolbox.aa_code import aa_mass
import pickle
import os
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sys import exit

# parser = argparse.ArgumentParser()
# parser.add_argument("--type", type=str, choices=["single", "double"])

# args = parser.parse_args()

with open("../HPS_Urry/data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

data = pd.read_table("../HPS_Urry/data/Rg.txt", header=0, sep="\s+")

rg_pc_single = []
rg_pc_double = []
rg_pc_3 = []
rg_exp = []
for idx_protein in range(data.shape[0]):
    protein_name = data.loc[idx_protein, "name"]
    mass = np.array([aa_mass[aa] for aa in seq[protein_name]])
    top = mdtraj.load_psf(f"../HPS_Urry/output/psf/{protein_name}.psf")
    T = 300

    if not os.path.exists(f"./output/traj_single_type/{protein_name}_{T:.2f}.dcd"):
        continue

    traj = mdtraj.load_dcd(
        f"./output/traj_single_type/{protein_name}_{T:.2f}.dcd", top=top
    )
    traj = traj[10000::10]
    rg = mdtraj.compute_rg(traj, masses=mass)
    rg_pc_single.append(rg.mean())

    # traj = mdtraj.load_dcd(
    #     f"./output/traj_double_type/{protein_name}_{T:.2f}.dcd", top=top
    # )
    # traj = traj[10000::10]
    # rg = mdtraj.compute_rg(traj, masses=mass)
    # rg_pc_double.append(rg.mean())

    with open(f"./output/Rg/{protein_name}_data.pkl", "rb") as f:
        rg = pickle.load(f)
        rg_pc_3.append(rg.mean())

    rg_exp.append(data.loc[idx_protein, "Rg_nm"])

    print(protein_name)

rg_pc_single = np.array(rg_pc_single)
#rg_pc_double = np.array(rg_pc_double)
rg_pc_3 = np.array(rg_pc_3)
rg_exp = np.array(rg_exp)

#rg_exp = data["Rg_nm"].values[0 : len(rg_pc_single)]

rg = pd.DataFrame({"exp": rg_exp, 
                   "pc_single": rg_pc_single, 
                   "pc_3": rg_pc_3})


with open(f'./output/params/train_protein_names.pkl', 'rb') as f:
    train_protein_names = pickle.load(f)

rg["train"] = False

for idx_protein in range(data.shape[0]):
    protein_name = data.loc[idx_protein, "name"]
    if protein_name in train_protein_names:
        rg.loc[idx_protein, "train"] = True
    else:
        rg.loc[idx_protein, "train"] = False

with open(f"./output/fig/rg.pkl", "wb") as f:
    pickle.dump(rg, f)
        
fig = plt.figure(figsize=(6, 6))
plt.clf()
ax = fig.add_subplot(111)
ax.plot(rg["exp"], rg["pc_single"], 'o', c="C0", label="PC single")
#ax.scatter(rg["exp"], rg["pc_double"], c="C1", label="PC double")
ax.plot(rg["exp"], rg["pc_3"], '^', c="C1", label="pc_3")
ax.plot([1, 6], [1, 6], c="k")
ax.set_xlabel("Experimental Rg (nm)")
ax.set_ylabel("Predicted Rg (nm)")
ax.legend()
plt.tight_layout()
os.makedirs("./output/fig", exist_ok=True)
plt.savefig("./output/fig/Rg.png", dpi=300)
