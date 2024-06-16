from functions import *
import pickle
import pandas as pd
import openmm as mm
import openmm.unit as unit
import openmm.app as app
import os
import time
from dltoolbox import *
from sys import exit

with open("./data/protein_seq.pkl", "rb") as f:
    seq = pickle.load(f)

data = pd.read_table('./data/Rg.txt', header=0, sep='\s+')

for idx_protein in range(data.shape[0]):
    protein_name = data.loc[idx_protein, 'name']
    topology = make_topology(seq[protein_name])
    os.makedirs('./output/psf', exist_ok=True)
    make_psf_from_topology(topology, f'./output/psf/{protein_name}.psf')

exit()
