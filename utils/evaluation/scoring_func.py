# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import numpy as np
from copy import deepcopy
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.QED import qed
from utils.evaluation.sascorer import compute_sa_score
# from utils.datasets import get_dataset
from rdkit.Chem.FilterCatalog import *
from collections import Counter


def is_pains(mol):
    params_pain = FilterCatalogParams()
    params_pain.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    catalog_pain = FilterCatalog(params_pain)
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    entry = catalog_pain.GetFirstMatch(mol)
    if entry is None:
        return False
    else:
        return True


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    all_rules = [int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]]
    return all_rules
    

def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight


def get_rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(mol)
    rmsd_list = []
    # predict 3d
    try:
        confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
        for confId in confIds:
            AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
            rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
            rmsd_list.append(rmsd)
        # mol3d = Chem.RemoveHs(mol3d)
        rmsd_list = np.array(rmsd_list)
        return [np.max(rmsd_list), np.min(rmsd_list), np.median(rmsd_list)]
    except:
        return [np.nan, np.nan, np.nan]


def get_logp(mol):
    return Crippen.MolLogP(mol)


def get_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    logp_score = get_logp(mol)
    lipinski_score = np.sum(obey_lipinski(mol))
    Chem.GetSymmSSSR(mol)
    ring_info = mol.GetRingInfo()
    ring_size = Counter([len(r) for r in ring_info.AtomRings()])
    # hacc_score = Lipinski.NumHAcceptors(mol)
    # hdon_score = Lipinski.NumHDonors(mol)

    return {
        'qed': qed_score,
        'sa': sa_score,
        'logp': logp_score,
        'lipinski': lipinski_score,
        'ring_size': ring_size
    }


def get_molecule_force_field(mol, conf_id=None, force_field='mmff', **kwargs):
    """
    Get a force field for a molecule.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    conf_id : int, optional
        ID of the conformer to associate with the force field.
    force_field : str, optional
        Force Field name.
    kwargs : dict, optional
        Keyword arguments for force field constructor.
    """
    if force_field == 'uff':
        ff = AllChem.UFFGetMoleculeForceField(
            mol, confId=conf_id, **kwargs)
    elif force_field.startswith('mmff'):
        AllChem.MMFFSanitizeMolecule(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(
            mol, mmffVariant=force_field)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, mmff_props, confId=conf_id, **kwargs)
    else:
        raise ValueError("Invalid force_field {}".format(force_field))
    return ff


def get_conformer_energies(mol, force_field='mmff'):
    """
    Calculate conformer energies.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    force_field : str, optional
        Force Field name.
    Returns
    -------
    energies : array_like
        Minimized conformer energies.
    """
    energies = []
    for conf in mol.GetConformers():
        ff = get_molecule_force_field(mol, conf_id=conf.GetId(), force_field=force_field)
        energy = ff.CalcEnergy()
        energies.append(energy)
    energies = np.asarray(energies, dtype=float)
    return energies


# class SimilarityWithMe:
#     def __init__(self, mol) -> None:
#         self.mol = deepcopy(mol)
#         self.mol = Chem.MolFromSmiles(Chem.MolToSmiles(self.mol))
#         self.fp= Chem.RDKFingerprint(self.mol)
#
#     def get_sim(self, mol):
#         mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize
#         fg_query = Chem.RDKFingerprint(mol)
#         sims = DataStructs.TanimotoSimilarity(self.fp, fg_query)
#         return sims
#
# class SimilarityWithTrain:
#     def __init__(self, base_dir='.') -> None:
#         self.cfg_dataset = EasyDict({
#             'name': 'pl',
#             'path': os.path.join(base_dir,  'data/crossdocked_pocket10'),
#             'split': os.path.join(base_dir, 'data/crossdocked_pocket10_split.pt'),
#             'fingerprint': os.path.join(base_dir, 'data/crossdocked_pocket10_fingerprint.pt'),
#             'smiles': os.path.join(base_dir, 'data/crossdocked_pocket10_smiles.pt'),
#         })
#         self.train_smiles = None
#         self.train_fingers = None
#
#     def _get_train_mols(self):
#         file_not_exists = (not os.path.exists(self.cfg_dataset.fingerprint)) or (not os.path.exists(self.cfg_dataset.smiles))
#         if file_not_exists:
#             _, subsets = get_dataset(config = self.cfg_dataset)
#             train_set = subsets['train']
#             self.train_smiles = []
#             self.train_fingers = []
#             for data in tqdm(train_set):  # calculate fingerprint and smiles of train data
#                 data.ligand_context_pos = data.ligand_pos
#                 data.ligand_context_element = data.ligand_element
#                 data.ligand_context_bond_index = data.ligand_bond_index
#                 data.ligand_context_bond_type = data.ligand_bond_type
#                 mol = reconstruct_from_generated_with_edges(data, sanitize=True)
#                 mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize
#                 smiles = Chem.MolToSmiles(mol)
#                 fg = Chem.RDKFingerprint(mol)
#                 self.train_fingers.append(fg)
#                 self.train_smiles.append(smiles)
#             self.train_smiles = np.array(self.train_smiles)
#             # self.train_fingers = np.array(self.train_fingers)
#             torch.save(self.train_smiles, self.cfg_dataset.smiles)
#             torch.save(self.train_fingers, self.cfg_dataset.fingerprint)
#         else:
#             self.train_smiles = torch.load(self.cfg_dataset.smiles)
#             self.train_fingers = torch.load(self.cfg_dataset.fingerprint)
#             self.train_smiles = np.array(self.train_smiles)
#             # self.train_fingers = np.array(self.train_fingers)
#
#     def _get_uni_mols(self):
#         self.train_uni_smiles, self.index_in_train = np.unique(self.train_smiles, return_index=True)
#         self.train_uni_fingers = [self.train_fingers[idx] for idx in self.index_in_train]
#
#     def get_similarity(self, mol):
#         if self.train_fingers is None:
#             self._get_train_mols()
#             self._get_uni_mols()
#         mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize
#         fp_mol = Chem.RDKFingerprint(mol)
#         # sim_func = DataStructs.TanimotoSimilarity
#         # with Pool(32) as pool:
#         #     sims = pool.map(partial(sim_func, bv1=fp_mol), self.train_uni_fingers)
#         sims = [DataStructs.TanimotoSimilarity(fp, fp_mol) for fp in self.train_uni_fingers]
#         return np.array(sims)
#
#
#     def get_top_sims(self, mol, top=3):
#         similarities = self.get_similarity(mol)
#         idx_sort = np.argsort(similarities)[::-1]
#         top_scores = similarities[idx_sort[:top]]
#         top_smiles = self.train_uni_smiles[idx_sort[:top]]
#         return top_scores, top_smiles


