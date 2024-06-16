import pandas as pd
import openmm as mm
import openmm.unit as unit
import openmm.app as app
import numpy as np
from dltoolbox.aa_code import aa_code_123, aa_code_321, aa_mass
from collections import defaultdict
from sys import exit

def make_system(seq, params, kappa):
    system = mm.System()

    ## add particles
    for aa in seq:
        if len(aa) == 1:
            aa = aa_code_123[aa]
        system.addParticle(params.loc[aa, "mass"])

    ## add harmonic bonds
    bond_force = mm.HarmonicBondForce()
    for i in range(len(seq) - 1):
        bond_force.addBond(i, i + 1, 0.38, 8033)
    bond_force.setForceGroup(1)
    system.addForce(bond_force)

    ## add electrostatics
    elec = mm.CustomNonbondedForce("138.935456 * q1 * q2 * exp(-kappa*r) / (D * r)")
    elec.addGlobalParameter("D", 80.0)
    elec.addGlobalParameter("kappa", kappa)
    elec.addPerParticleParameter("q")
    for aa in seq:
        if len(aa) == 1:
            aa = aa_code_123[aa]
        elec.addParticle([params.loc[aa, "charge"]])
    elec.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    elec.setCutoffDistance(4.0 * unit.nanometer)
    elec.createExclusionsFromBonds([[i, i + 1] for i in range(len(seq) - 1)], 1)
    elec.setForceGroup(2)
    system.addForce(elec)

    ## add nonbonded interactions
    hps_formula = [
        "step(r_cut - r) * (u_lj + epsilon*(1 - lambda)) + step(r - r_cut)*lambda*u_lj",
        "r_cut = 2^(1./6)*sigma",
        "u_lj = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)",
        "sigma = 0.5*(sigma1 + sigma2)",
        "lambda = 0.5*(lambda1 + lambda2)",
        "epsilon = 0.8368",
    ]
    hps = mm.CustomNonbondedForce(";".join(hps_formula))
    hps.addPerParticleParameter("sigma")
    hps.addPerParticleParameter("lambda")

    for aa in seq:
        if len(aa) == 1:
            aa = aa_code_123[aa]
        hps.addParticle([params.loc[aa, "sigma"], params.loc[aa, "lambda"]])

    hps.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
    hps.setCutoffDistance(4.0 * unit.nanometer)
    hps.createExclusionsFromBonds([[i, i + 1] for i in range(len(seq) - 1)], 1)
    hps.setForceGroup(3)
    system.addForce(hps)

    system.addForce(mm.CMMotionRemover())

    return system

def make_topology(seq):
    topology = app.Topology()
    chain = topology.addChain()
    for aa in seq:
        if len(aa) == 1:
            aa = aa_code_123[aa]
        residue = topology.addResidue(aa, chain)
        topology.addAtom("CA", app.element.carbon, residue)

    atoms = list(topology.atoms())
    for i in range(len(seq) - 1):
        topology.addBond(atoms[i], atoms[i + 1])  
    
    return topology

def make_initial_position(seq):
    position = np.zeros((len(seq), 3))
    for i in range(len(seq)):
        position[i] = [i*0.38, 0, 0]
    return position