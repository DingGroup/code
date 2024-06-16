import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sys import exit
import numpy as np
from dltoolbox.aa_code import aa_code_321, aa_code_123
from tabulate import tabulate

params = pd.read_csv(
    "../HPS_Urry/data/params_hps_Urry.csv", header=0, index_col=0,
)
names = params.index.values
lambda_init = params['lambda'].values
index = np.argsort(lambda_init)

names = names[index]
names = [aa_code_321[i] for i in names]
lambda_init = lambda_init[index]


lambda_pc = []
for i in range(5):
    with open(f'../HPS_Urry_PC_{i}/output/params/single_type_params_optimized.pkl', 'rb') as file_handle:
        tmp = pickle.load(file_handle)
    lambda_pc.append(tmp[index])
fig = plt.figure(0)
fig.clf()
plt.plot(range(20), lambda_init,  'o', markerfacecolor='none',label='HPS-Urry')
plt.plot(range(20), lambda_pc[0], '^', markerfacecolor='none',label='Iteration 1')
plt.plot(range(20), lambda_pc[1], 'v', markerfacecolor='none', label='Iteration 2')
plt.plot(range(20), lambda_pc[2], 's', markerfacecolor='none', label='Iteration 3')
plt.plot(range(20), lambda_pc[3], 'p', markerfacecolor='none', label='Iteration 4')         
plt.plot(range(20), lambda_pc[4], 'x', markerfacecolor='none', label='Iteration 5')         
plt.legend(ncol=2)
plt.xlabel('Amino acids')
plt.ylabel(r'Hydrophobicity scale $\lambda$')
plt.xticks(range(20), names)
plt.ylim(-0.1, 1.3)
plt.yticks(np.arange(0, 1.1, 0.2))
plt.tight_layout()
plt.savefig('./output/fig/lambdas.png')
plt.savefig('./output/fig/lambdas.eps')

# Table

with open("../HPS_Urry/data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

with open("../HPS_Urry_PC_0/output/params/train_protein_names.pkl", 'rb') as file_handle:
    train_protein_names = pickle.load(file_handle)

train_seqs = [seq[n] for n in train_protein_names]
train_seq = ''.join(train_seqs)

aa_freq = [ train_seq.count(i)/len(train_seq) for i in names]


names = [aa_code_123[i] for i in names]
print(tabulate({'Amino acid': names, 'Freq': aa_freq, 'HPS-Urry': lambda_init, 'Iteration 1': lambda_pc[0], 'Iteration 2': lambda_pc[1], 'Iteration 3': lambda_pc[2], 'Iteration 4': lambda_pc[3], 'Iteration 5': lambda_pc[4]}, headers='keys', tablefmt='latex', floatfmt=('.3f', '.3f', '.2f', '.2f', '.2f', '.2f', '.2f', '.2f')))
                                                                                                                                                                                                                                                            