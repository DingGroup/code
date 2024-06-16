import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sys import exit
import random

with open("./output/fig/rg.pkl", "rb") as f:
    rg = pickle.load(f)


rg_train = rg[rg["train"] == True]
rg_test = rg[rg["train"] == False]

wc_list = [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

r_train_urry = stats.pearsonr(rg_train["exp"], rg_train["urry"])[0]
rmse_train_urry = ((rg_train["exp"] - rg_train["urry"])**2).mean()**0.5

r_train_pc = []
rmse_train_pc = []
for wc in wc_list:
    r_train_pc.append(stats.pearsonr(rg_train["exp"], rg_train[f"pc_{wc:.2E}"])[0])
    rmse_train_pc.append(((rg_train["exp"] - rg_train[f"pc_{wc:.2E}"])**2).mean()**0.5)

r_test_urry = stats.pearsonr(rg_test["exp"], rg_test["urry"])[0]
rmse_test_urry = ((rg_test["exp"] - rg_test["urry"])**2).mean()**0.5
r_test_pc = []
rmse_test_pc = []
for wc in wc_list:
    r_test_pc.append(stats.pearsonr(rg_test["exp"], rg_test[f"pc_{wc:.2E}"])[0])
    rmse_test_pc.append(((rg_test["exp"] - rg_test[f"pc_{wc:.2E}"])**2).mean()**0.5)

data_train = rg[rg["train"] == True]
data_test = rg[rg["train"] == False]

exit()

for data, label in zip([data_train, data_test], ['train', 'test']):
    rg_exp = data['exp']
    rg_sim = data['pc_1.00E-04']
    pearsonr, _ = stats.pearsonr(rg_exp, rg_sim)
    rmse = ((rg_exp - rg_sim)**2).mean()**0.5

    fig = plt.figure()
    plt.plot(rg_exp, rg_sim, 'o', color = 'blue', markersize=8, markerfacecolor='none') 
    plt.xlim(1, 5.5)
    plt.ylim(1, 5.5)
    plt.xticks(ticks = [1, 2, 3, 4, 5])
    plt.xticks(ticks = [1, 2, 3, 4, 5])
    plt.xlabel(r'Experimental $R_g$ (nm)')
    plt.ylabel(r'Simulated $R_g$ (nm)')
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot([0, 6], [0, 6], '--', color = 'black', linewidth=0.8)
    plt.text(1.2, 4.5, f'Pearson r = {pearsonr:.2f}\nRMSE = {rmse:.2f}', fontsize=16)

    plt.tight_layout()
    plt.savefig(f'./output/fig/Rg_{label}.eps')
    plt.savefig(f'./output/fig/Rg_{label}.png')
