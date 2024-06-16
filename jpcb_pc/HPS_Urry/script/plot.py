import pickle
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

with open("./output/fig/rg_sim.pkl", "rb") as f:
    rg_sim = np.array(pickle.load(f))

data = pd.read_table("./data/Rg.txt", header=0, sep="\s+")
names = data["name"].values

np.random.seed(100)
names_train = np.random.choice(names, size=15, replace=False)

train_idx = []
test_idx = []
for i in range(len(names)):
    if names[i] in names_train:
        train_idx.append(i)
    else:
        test_idx.append(i)

pearsonr, _ = stats.pearsonr(data["Rg_nm"], rg_sim)
pearsonr_train, _ = stats.pearsonr(data.loc[train_idx, "Rg_nm"], rg_sim[train_idx])
pearsonr_test, _ = stats.pearsonr(data.loc[test_idx, "Rg_nm"], rg_sim[test_idx])


rmse = ((data["Rg_nm"] - rg_sim) ** 2).mean() ** 0.5
rmse_train = ((data.loc[train_idx, "Rg_nm"] - rg_sim[train_idx]) ** 2).mean() ** 0.5
rmse_test = ((data.loc[test_idx, "Rg_nm"] - rg_sim[test_idx]) ** 2).mean() ** 0.5


fig = plt.figure()
# plt.plot(data['Rg_nm'], rg_sim, 'o', color = 'blue', markersize=8, markerfacecolor='none')
plt.plot(
    data.loc[train_idx, "Rg_nm"],
    rg_sim[train_idx],
    "o",
    color="blue",
    markersize=8,
    markerfacecolor="none",
)

plt.xlim(1, 5.5)
plt.ylim(1, 5.5)
plt.xticks(ticks=[1, 2, 3, 4, 5])
plt.xticks(ticks=[1, 2, 3, 4, 5])
plt.xlabel(r"Experimental $R_g$ (nm)")
plt.ylabel(r"Simulated $R_g$ (nm)")
ax = plt.gca()
ax.set_aspect(1)
plt.plot([0, 6], [0, 6], "--", color="black", linewidth=0.8)
plt.text(1.2, 4.5, f"Pearson r = {pearsonr_train:.2f}\nRMSE = {rmse_train:.2f}", fontsize=16)


plt.tight_layout()
plt.savefig(f"./output/fig/Rg_train.eps")
plt.savefig(f"./output/fig/Rg_train.png")

fig = plt.figure()
plt.plot(
    data.loc[test_idx, "Rg_nm"],
    rg_sim[test_idx],
    "o",
    color="blue",
    markersize=8,
    markerfacecolor="none",
)

plt.xlim(1, 5.5)
plt.ylim(1, 5.5)
plt.xticks(ticks=[1, 2, 3, 4, 5])
plt.xticks(ticks=[1, 2, 3, 4, 5])
plt.xlabel(r"Experimental $R_g$ (nm)")
plt.ylabel(r"Simulated $R_g$ (nm)")
ax = plt.gca()
ax.set_aspect(1)
plt.plot([0, 6], [0, 6], "--", color="black", linewidth=0.8)
plt.text(1.2, 4.5, f"Pearson r = {pearsonr_test:.2f}\nRMSE = {rmse_test:.2f}", fontsize=16)


plt.tight_layout()
plt.savefig(f"./output/fig/Rg_test.eps")
plt.savefig(f"./output/fig/Rg_test.png")