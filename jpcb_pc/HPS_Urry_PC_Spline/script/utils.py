import pickle
import pandas as pd
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from dltoolbox.aa_code import aa_code_123, aa_code_321

# with open(f"./output/params/double_type_params_optimized.pkl", "rb") as f:
#     lamb = pickle.load(f)

params = pd.read_csv(
    "../HPS_Urry/data/params_hps_Urry.csv",
    header=0,
    index_col=0,
)

def make_hps_spline(seq, params, unb):
    func = mm.Continuous3DFunction(
        xsize=unb.shape[0],
        ysize=unb.shape[1],
        zsize=unb.shape[2],
        values=unb.reshape(-1, order="F"),
        xmin=0.0,
        xmax=float(unb.shape[0] - 1),
        ymin=0.0,
        ymax=float(unb.shape[1] - 1),
        zmin=0.0,
        zmax=2.0,
    )

    fnb = mm.CustomNonbondedForce(
        "4*eps*((sigma/r)^12) + unb(t1, t2, r);sigma= 0.5*(sigma1+sigma2);"
    )
    fnb.addGlobalParameter("eps", 0.8368)
    fnb.addTabulatedFunction("unb", func)
    fnb.addPerParticleParameter("t")
    fnb.addPerParticleParameter("sigma")

    for aa in seq:
        if len(aa) == 1:
            aa = aa_code_123[aa]
        fnb.addParticle([list(params.index).index(aa), params.loc[aa, "sigma"]])

    fnb.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
    fnb.setCutoffDistance(4.0 * unit.nanometer)
    fnb.createExclusionsFromBonds([[i, i + 1] for i in range(len(seq) - 1)], 1)
    fnb.setForceGroup(4)
    return fnb

def make_hps_double_type(seq, params, lamb):
    lamb_array = np.zeros((params.shape[0], params.shape[0]))
    k = 0
    for i in range(params.shape[0]):
        for j in range(i, params.shape[0]):
            lamb_array[i, j] = lamb[k]
            lamb_array[j, i] = lamb[k]
            k += 1

    f_lamb = mm.Discrete2DFunction(
        params.shape[0], params.shape[0], lamb_array.T.flatten().tolist()
    )

    ## add nonbonded interactions
    hps_formula = [
        "step(r_cut - r) * (u_lj + epsilon*(1 - lambda)) + step(r - r_cut)*lambda*u_lj",
        "r_cut = 2^(1./6)*sigma",
        "u_lj = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)",
        "sigma = 0.5*(sigma1 + sigma2)",
        "lambda = f_lamb(idx1, idx2)",
        "epsilon = 0.8368",
    ]
    hps = mm.CustomNonbondedForce(";".join(hps_formula))
    hps.addTabulatedFunction("f_lamb", f_lamb)
    hps.addPerParticleParameter("sigma")
    hps.addPerParticleParameter("idx")

    for aa in seq:
        if len(aa) == 1:
            aa = aa_code_123[aa]
        hps.addParticle([params.loc[aa, "sigma"], list(params.index).index(aa)])

    hps.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
    hps.setCutoffDistance(4.0 * unit.nanometer)
    hps.createExclusionsFromBonds([[i, i + 1] for i in range(len(seq) - 1)], 1)
    hps.setForceGroup(3)

    return hps