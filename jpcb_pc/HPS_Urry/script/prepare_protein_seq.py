import pickle
from sys import exit

protein_seq = {}
with open("./data/sequences.fasta", "r") as f:
    for line in f:
        if line.startswith(">"):
            protein_id = line.strip()[1:]
            protein_seq[protein_id] = ""
        else:
            protein_seq[protein_id] += line.strip()

with open("./data/protein_seq.pkl", "wb") as f:
    pickle.dump(protein_seq, f)