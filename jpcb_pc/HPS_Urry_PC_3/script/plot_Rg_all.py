import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sys import exit
import numpy as np

with open("../HPS_Urry/output/fig/rg_sim.pkl", "rb") as f:
    rg_urry = pickle.load(f)


with open("./output/fig/rg.pkl", "rb") as f:
    data = pickle.load(f)

rg_exp = data['exp']
rg_pc3 = data['pc_single']

fig = plt.figure()
plt.clf()
plt.plot(rg_exp, rg_urry, '^', label='Initial force field', markersize=8, markerfacecolor='none' )
plt.plot(rg_exp, rg_pc3, 'o', label='Optimized force field', markersize=8, markerfacecolor='none')

ax = plt.gca()
ax.set_aspect(1)
plt.plot([0, 6], [0, 6], '--', color = 'black')
plt.xlim(0.8, 5.8)
plt.ylim(0.8, 5.8)
plt.xticks(np.arange(1, 6, 1))
plt.xlabel(r'Experimental $R_g$ (nm)')
plt.ylabel(r'Simulated $R_g$ (nm)')
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('./output/fig/rg_urry_vs_pc3.png')
plt.savefig('./output/fig/rg_urry_vs_pc3.eps')