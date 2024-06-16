import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/xqding/my_projects_on_github/pccg")
from pccg.utils import spline
import pandas as pd
import os
import matplotlib.cm as cm
from sys import exit

wc = 1e-4
with open(f'./output/params/params_optimized_wc_{wc:.2E}.pkl', 'rb') as f:
    theta = pickle.load(f)

tmp = np.zeros((20, 20, theta.shape[-1]))
idx = 0
for i in range(20):
    for j in range(i, 20):
        tmp[i, j] = theta[idx]
        tmp[j, i] = theta[idx]
        idx += 1
theta = tmp        

params = pd.read_csv(
    "../HPS_Urry/data/params_hps_Urry.csv",
    header=0,
    index_col=0,
)

aa_names = params.index.values

ncols = 4
nrows = 6
fig = plt.figure(figsize = (6.4*ncols, 4.8*nrows))
plt.clf()
for i in range(len(aa_names)):
    aa_i = aa_names[i]
    print(aa_i)
    ax = plt.subplot(nrows, ncols, i + 1)
    colors = iter([cm.tab20(k) for k in range(20)])

    lines = []
    for j in range(len(aa_names)):
        aa_j = aa_names[j]

        sigma_i, sigma_j = params.loc[aa_i, "sigma"], params.loc[aa_j, "sigma"]
        sigma = 0.5 * (sigma_i + sigma_j)
        lamb_i, lamb_j = params.loc[aa_i, "lambda"], params.loc[aa_j, "lambda"]
        lamb = 0.5 * (lamb_i + lamb_j)
        eps = 0.8368

        r = np.linspace(sigma - 0.02, 2.0, 1000)
        u_lj = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
        r0 = 2 ** (1.0 / 6) * sigma
        u_hps = np.heaviside(r0 - r, 0) * (u_lj + eps * (1 - lamb))
        u_hps += np.heaviside(r - r0, 0) * lamb * u_lj

        u_rep = 4 * eps * ((sigma / r) ** 12)

        basis = spline.bs_nb(r, 2.0, 12)
        u_spline = np.matmul(basis, theta[i,j])
        u_spline += u_rep
        line = ax.plot(r, u_spline, linewidth=2.0, label=aa_j, color=next(colors))
        lines.append(line)

    plt.ylim(-2.0, 5.0)
    # plt.xlim(0, lj_ff['max'])
    plt.title(aa_i)
    # plt.text(0.1, 3, aa_i)
    plt.xlabel('Distance (nm)')
    plt.ylabel('Energy (kJ/mol)')

ax = plt.subplot(nrows, ncols, 21)
colors = iter([cm.tab20(i) for i in range(20)])
# ax.legend(lines, AA_names)
for aa_j in aa_names:
    plt.plot([], [],
            linewidth = 2.0,
            label = aa_j,
            color = next(colors))
ax.legend(ncol = 4, loc = 'upper center') 
ax.axis('off')
plt.tight_layout()
plt.savefig(f"./output/fig/spline.png")
plt.savefig(f"./output/fig/spline.eps")


fig = plt.figure()
plt.clf()
pairs = [('ALA','TRP'), ('PHE', 'ARG'), ('ASP', 'THR'), ('TYR', 'GLN')]
for i, (aa_i, aa_j) in enumerate(pairs):
    sigma_i, sigma_j = params.loc[aa_i, "sigma"], params.loc[aa_j, "sigma"]
    sigma = 0.5 * (sigma_i + sigma_j)
    lamb_i, lamb_j = params.loc[aa_i, "lambda"], params.loc[aa_j, "lambda"]
    lamb = 0.5 * (lamb_i + lamb_j)
    eps = 0.8368

    r = np.linspace(sigma - 0.02, 2.0, 1000)
    u_lj = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    r0 = 2 ** (1.0 / 6) * sigma
    u_hps = np.heaviside(r0 - r, 0) * (u_lj + eps * (1 - lamb))
    u_hps += np.heaviside(r - r0, 0) * lamb * u_lj

    u_rep = 4 * eps * ((sigma / r) ** 12)

    basis = spline.bs_nb(r, 2.0, 12)

    u_spline = np.matmul(basis, theta[list(aa_names).index(aa_i), list(aa_names).index(aa_j)])
    u_spline += u_rep

    plt.plot(r, u_spline, label=f"{aa_i}-{aa_j}")

plt.xlabel('Distance (nm)')
plt.ylabel('Energy (kJ/mol)')
plt.legend()
plt.tight_layout()
plt.savefig(f"./output/fig/spline_pairs.png")
plt.savefig(f"./output/fig/spline_pairs.eps")

exit()


k = 0
fig = plt.figure(figsize=(6.4*5, 4.8*42))
plt.clf()
for i in range(params.shape[0]):
    for j in range(i, params.shape[0]):        
        aa_i, aa_j = params.index[i], params.index[j]
        sigma_i, sigma_j = params.loc[aa_i, "sigma"], params.loc[aa_j, "sigma"]
        sigma = 0.5 * (sigma_i + sigma_j)
        lamb_i, lamb_j = params.loc[aa_i, "lambda"], params.loc[aa_j, "lambda"]
        lamb = 0.5 * (lamb_i + lamb_j)
        eps = 0.8368

        r = np.linspace(sigma - 0.02, 2.0, 1000)
        u_lj = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
        r0 = 2 ** (1.0 / 6) * sigma
        u_hps = np.heaviside(r0 - r, 0) * (u_lj + eps * (1 - lamb))
        u_hps += np.heaviside(r - r0, 0) * lamb * u_lj

        u_rep = 4 * eps * ((sigma / r) ** 12)

        basis = spline.bs_nb(r, 2.0, 12)



        plt.subplot(42, 5, k+1)
        plt.plot(r, u_hps, '--', label="HPS")
        for wc in wc_list:
            u_spline = np.matmul(basis, theta[wc][k])
            u_spline += u_rep
            plt.plot(r, u_spline, label=f"wc_{wc:.2E}")
        
        if k == 0:
            plt.legend()

        k += 1

    print(i)

os.makedirs("./output/fig", exist_ok=True)
plt.savefig(f"./output/fig/spline.png")
