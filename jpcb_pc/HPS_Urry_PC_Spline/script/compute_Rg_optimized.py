import mdtraj
import pandas as pd
import numpy as np
from dltoolbox.aa_code import aa_mass
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats

# parser = argparse.ArgumentParser()
# parser.add_argument("--type", type=str, choices=["single", "double"])

# args = parser.parse_args()

with open("../HPS_Urry/data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

data = pd.read_table("../HPS_Urry/data/Rg.txt", header=0, sep="\s+")

#wc_list = [0.0, 1e-10, 1e-5, 1e-4, 1e-3]
wc_list = [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

rg_pc = defaultdict(list)
rg_urry = []
rg_exp = []
for idx_protein in range(data.shape[0]):
    protein_name = data.loc[idx_protein, "name"]
    mass = np.array([aa_mass[aa] for aa in seq[protein_name]])
    top = mdtraj.load_psf(f"../HPS_Urry/output/psf/{protein_name}.psf")
    T = 300

    for wc in wc_list:    
        if not os.path.exists(f"./output/traj_wc_{wc:.2E}/{protein_name}_{T:.2f}.dcd"):
            continue

        traj = mdtraj.load_dcd(
            f"./output/traj_wc_{wc:.2E}/{protein_name}_{T:.2f}.dcd", top=top
        )
        traj = traj[10000::10]
        rg = mdtraj.compute_rg(traj, masses=mass)
        rg_pc[wc].append(rg.mean())

    with open(f"./output/Rg/{protein_name}_data.pkl", "rb") as f:
        rg = pickle.load(f)
        rg_urry.append(rg.mean())

    rg_exp.append(data.loc[idx_protein, "Rg_nm"])

    print(protein_name)

for wc in wc_list:
    rg_pc[wc] = np.array(rg_pc[wc])
rg_urry = np.array(rg_urry)
rg_exp = np.array(rg_exp)

rg = pd.DataFrame({"exp": rg_exp,  
                   "urry": rg_urry})
for wc in wc_list:
    rg[f"pc_{wc:.2E}"] = rg_pc[wc]

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
for wc in wc_list:
    ax.plot(rg["exp"], rg[f"pc_{wc:.2E}"], 'o', label=f"PC {wc:.2E}")
ax.plot(rg["exp"], rg["urry"], '^', label="Urry")
ax.plot([1, 6], [1, 6], c="k")
ax.set_xlabel("Experimental Rg (nm)")
ax.set_ylabel("Predicted Rg (nm)")
ax.legend()
plt.tight_layout()
plt.savefig("./output/fig/Rg.png", dpi=300)

rg_train = rg[rg["train"]]
rg_test = rg[~rg["train"]]

r_train_urry = stats.pearsonr(rg_train["exp"], rg_train["urry"])[0]
r_train_pc = []
for wc in wc_list:
    r_train_pc.append(stats.pearsonr(rg_train["exp"], rg_train[f"pc_{wc:.2E}"])[0])

r_test_urry = stats.pearsonr(rg_test["exp"], rg_test["urry"])[0]
r_test_pc = []
for wc in wc_list:
    r_test_pc.append(stats.pearsonr(rg_test["exp"], rg_test[f"pc_{wc:.2E}"])[0])
