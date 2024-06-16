import pandas as pd
import pickle
import tabulate

data = pd.read_table("../HPS_Urry/data/Rg.txt", header=0, sep="\s+")

with open(f'./output/params/train_protein_names.pkl', 'rb') as f:
    train_protein_names = pickle.load(f)

data["label"] = ''

for idx_protein in range(data.shape[0]):
    protein_name = data.loc[idx_protein, "name"]
    if protein_name in train_protein_names:
        data.loc[idx_protein, "label"] = 'train'
    else:
        data.loc[idx_protein, "label"] = 'test'

data = data.sort_values(by='label', ascending=False)


headers = ['name', 'Rg_nm', 'label']
values = data[headers].values.tolist()

print(tabulate.tabulate(values, headers=headers, tablefmt='latex', floatfmt = '.2f'))

