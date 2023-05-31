# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.Chem import rdMolAlign
from copy import deepcopy


def break_rotatable_bond(mol, rotatable_bond=None):
    """Break a single mol into multiple fragmetned mols by rotatable bonds."""
    if rotatable_bond is None:
        rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
        rotatable_bond = []
        for atom_1, atom_2 in rot_atom_pairs:
            rotatable_bond.append(mol.GetBondBetweenAtoms(atom_1, atom_2).GetIdx())
    if len(rotatable_bond) == 0:
        return [mol], []
    f_mol = Chem.FragmentOnBonds(mol, rotatable_bond)
    f_mol = Chem.GetMolFrags(f_mol, asMols=True)
    return f_mol, rotatable_bond


def get_num_rotatable_bonds(mol):
    rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
    return len(rot_atom_pairs)


# rdmol conformer
def compute_3d_coors(mol, random_seed=0):
    mol = Chem.AddHs(mol)
    success = AllChem.EmbedMolecule(mol, randomSeed=random_seed)
    if success == -1:
        return 0, 0
    mol = Chem.RemoveHs(mol)
    c = mol.GetConformer(0)
    pos = c.GetPositions()
    return pos, 1


def compute_3d_coors_multiple(mol, numConfs=20, maxIters=400, randomSeed=1):
    # mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol, addCoords=True)
    AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, numThreads=0, randomSeed=randomSeed)
    if mol.GetConformers() == ():
        return None, [], 0
    try:
        result = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=maxIters, numThreads=0)
    except Exception as e:
        print(str(e))
        return None, [], 0
    mol = Chem.RemoveHs(mol)
    result = [tuple((result[i][0], result[i][1], i)) for i in range(len(result)) if result[i][0] == 0]
    if result == []:  # no local minimum on energy surface is found
        return None, [], 0
    result.sort()
    # return mol.GetConformers()[result[0][-1]].GetPositions(), 1
    return mol, result, 1


def get_rmsd(ref, pred, heavy_only=True):
    """Calculate RMSD between two conformers."""
    if heavy_only:
        ref = Chem.RemoveHs(ref)
        pred = Chem.RemoveHs(pred)
    return rdMolAlign.GetBestRMS(pred, ref)


def ff_optimize(ori_mol, addHs=False, enable_torsion=False):
    mol = deepcopy(ori_mol)
    Chem.GetSymmSSSR(mol)
    if addHs:
        mol = Chem.AddHs(mol, addCoords=True)
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    if mp is None:
        return (None,)

    #     # turn off angle-related terms
    #     mp.SetMMFFOopTerm(enable_torsion)
    #     mp.SetMMFFAngleTerm(True)
    #     mp.SetMMFFTorsionTerm(enable_torsion)

    #     # optimize unrelated to angles
    #     mp.SetMMFFStretchBendTerm(True)
    #     mp.SetMMFFBondTerm(True)
    #     mp.SetMMFFVdWTerm(True)
    #     mp.SetMMFFEleTerm(True)

    try:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        energy_before_ff = ff.CalcEnergy()
        grad = ff.CalcGrad()
        # print(ff.CalcGrad())
        ff.Minimize()
        energy_after_ff = ff.CalcEnergy()
        # print('After: ', ff.CalcGrad())
        # print(f'Energy: {energy_before_ff} --> {energy_after_ff}')
        energy_change = energy_before_ff - energy_after_ff
        Chem.SanitizeMol(ori_mol)
        Chem.SanitizeMol(mol)
        rmsd = rdMolAlign.GetBestRMS(ori_mol, mol)
    except:
        return (None,)
    mol = Chem.RemoveHs(mol)
    return energy_change, rmsd, mol


def get_ring_systems(mol, includeSpiro=False):
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon>1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
    systems = tuple(tuple(i) for i in systems)
    return systems


def num_x_mem_ring(mol, ring_sizes):
    counts = [0 for _ in range(len(ring_sizes))]
    # ri = mol.GetRingInfo()
    # single_rings = ri.AtomRings()
    fused_rings = get_ring_systems(mol)
    # rings = single_rings + fused_rings
    for ringAts in fused_rings:
        ring_size = len(ringAts)
        if ring_size in ring_sizes:
            ind = ring_sizes.index(ring_size)
            counts[ind] += 1
    return counts
