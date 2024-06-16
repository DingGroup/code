import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_table('./data/Rg.txt', header=0, sep='\s+')

idx_protein = 1

protein_name = data.loc[idx_protein, 'name']
with open(f"./output/traj/{protein_name}_TRE.pkl", "rb") as f:
    data = pickle.load(f)
record = np.array(data['record'])
accept_rate = data['accept_rate']

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
for j in range(6):
    ax.plot(record[-100:,j], label = f'{j}')
ax.legend()
ax.set_xlabel('iteration')
plt.savefig(f'./output/fig/{protein_name}_TRE.png')


