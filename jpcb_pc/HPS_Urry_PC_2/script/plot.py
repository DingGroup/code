import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sys import exit

with open("./output/fig/rg.pkl", "rb") as f:
    data = pickle.load(f)

data_train = data[data['train'] == True]
data_test = data[data['train'] == False]


for data, label in zip([data_train, data_test], ['train', 'test']):
    rg_exp = data['exp']
    rg_sim = data['pc_single']
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
