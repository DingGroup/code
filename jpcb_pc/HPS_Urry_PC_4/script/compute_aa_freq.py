import pickle
import numpy as np
import pandas as pd
from dltoolbox.aa_code import aa_code_123, aa_code_321
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

with open("../HPS_Urry/data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)


data = pd.read_table("../HPS_Urry/data/Rg.txt", header=0, sep="\s+")
names = data["name"].values    

np.random.seed(100)
names = np.random.choice(names, size = 15, replace=False)

train_seqs = [seq[n] for n in names]
seq = ''.join(train_seqs)


params = pd.read_csv(
    "../HPS_Urry/data/params_hps_Urry.csv", header=0, index_col=0,
)
names = params.index.values
lambda_init = params['lambda'].values
index = np.argsort(lambda_init)
names = names[index]
lambda_init = lambda_init[index]


aa_freq = [ seq.count(aa_code_321[i])/len(seq) for i in names]

aa_protein_freq = []
for i in names:
    j = aa_code_321[i]
    aa_protein_freq.append(np.mean([j in s for s in train_seqs]))


with open(f'../HPS_Urry_PC_0/output/params/single_type_params_optimized.pkl', 'rb') as file_handle:
    tmp = pickle.load(file_handle)
lambda_pc = tmp[index]

lambda_diff = np.abs(lambda_pc - lambda_init)


fig = plt.figure(0)
plt.clf()
plt.plot(aa_freq, lambda_diff, 'o')
plt.xlabel('Amino acid frequency')
plt.ylabel(r'$|\lambda - \lambda_{\rm init}|$')
plt.tight_layout()
plt.savefig('./output/fig/aa_freq_vs_lambda_diff.png')
