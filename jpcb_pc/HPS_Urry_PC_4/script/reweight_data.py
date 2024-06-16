#!/home/xqding/apps/miniconda3/envs/xd/bin/python

#SBATCH --job-name=ME
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-24
#SBATCH --output=./slurm_output/ME_%a.out
#SBATCH --open-mode=truncate

import pickle
import numpy as np
import cvxpy as cp
import os
import sys
import pandas as pd
from sys import exit

data = pd.read_table('../HPS_Urry/data/Rg.txt', header=0, sep='\s+')

idx_protein = int(os.environ['SLURM_ARRAY_TASK_ID'])

protein_name = data.loc[idx_protein, 'name']

with open(f'./output/Rg/{protein_name}_data.pkl', 'rb') as f:
    Rg = pickle.load(f)

Rg_exp = data.loc[idx_protein, 'Rg_nm']    

q = cp.Variable(len(Rg), nonneg=True)
objective = cp.Maximize(cp.sum(cp.entr(q)))
constraints = [cp.sum(q) == 1, cp.sum(cp.multiply(q, Rg)) == Rg_exp]
prob = cp.Problem(objective, constraints)
prob.solve(solver = 'SCS', verbose = True)

# prob.solve(solver = 'CLARABEL', verbose = True)
## THe CLARABLE solver is faster than SCS, but it fails or gives inaccurate results for some proteins. The SCS solver is slower but more robust, so we use it here.

q = q.value
q = q / np.sum(q)

os.makedirs('./output/ME', exist_ok=True)
with open(f'./output/ME/{protein_name}_weight.pkl', 'wb') as f:
    pickle.dump(q, f)

print(f'{protein_name} is done!')