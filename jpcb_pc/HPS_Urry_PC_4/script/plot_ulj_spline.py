import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/xqding/my_projects_on_github/pccg")
from pccg.utils import spline
import pandas as pd
import os

with open(f'./output/params/spline_params_optimized.pkl', 'rb') as f:
    theta = pickle.load(f)

params = pd.read_csv(
    "../HPS_Urry/data/params_hps_Urry.csv",
    header=0,
    index_col=0,
)

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

        knots = np.linspace(0, 2.0, 13)
        boundary_knots = knots[[0, -1]]
        knots = knots[1:-1]
        basis = spline.bs(r, knots, boundary_knots)

        u_spline = np.matmul(basis, theta[k])
        u_spline += u_rep

        plt.subplot(42, 5, k+1)
        plt.plot(r, u_hps, '--', label="HPS")
        plt.plot(r, u_spline, label="spline")
        plt.legend()

        k += 1

    print(i)

os.makedirs("./output/fig", exist_ok=True)
plt.savefig(f"./output/fig/spline.png")
    

