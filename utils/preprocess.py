# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import copy
import itertools
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from utils.misc import DecomposeError


def decompose_molecule(mol, method='BRICS'):
    mol = copy.deepcopy(mol)
    if method == 'BRICS':
        raw_frags_smiles = BRICS.BRICSDecompose(mol)
        frags_smiles, frags_atom_idx = [], []
        # print('raw frag smiles: ', raw_frags_smiles)
        # print('raw smiles: ', Chem.MolToSmiles(mol))
        # for atom in mol.GetAtoms():
        #     atom.SetIsotope(0)
        # print('remove isotope')
        for smiles in list(raw_frags_smiles):
            # remove dummy atom, replace with H to avoid Kekulize error in aromatic rings
            rogue_smiles = re.sub('\[[0-9]+\*\]', '[H]', smiles)
            rogue_smiles = re.sub('\(\)', '', rogue_smiles)
            rogue_frag = Chem.MolFromSmiles(rogue_smiles)
            # print(smiles, rogue_smiles)
            # for atom in rogue_frag.GetAtoms():
            #     atom.SetIsotope(0)
            assert mol.HasSubstructMatch(rogue_frag)
            frags_smiles.append(rogue_smiles)
            frags_atom_idx.append(mol.GetSubstructMatches(rogue_frag))

        # # if we want to include dummy atoms
        # frags = list(BRICS.BRICSDecompose(mol, returnMols=True))
        # for i, f in enumerate(frags):
        #     for atom in f.GetAtoms():
        #         atom.SetIsotope(0)
        #     params = Chem.AdjustQueryParameters()
        #     # params.makeBondsGeneric = True
        #     params.makeDummiesQueries = True
        #     rogue_frag = Chem.AdjustQueryProperties(f, params)
        #     print("(%d)"%i, Chem.MolToSmiles(rogue_frag), mol.HasSubstructMatch(rogue_frag))
        #     print(mol.GetSubstructMatches(rogue_frag))

    # from rdkit.Chem.Scaffolds import MurckoScaffold
    # core = MurckoScaffold.GetScaffoldForMol(mol)
    # # Chem.MolToSmiles(core)
    # core.RemoveAllConformers()
    # core
    else:
        raise NotImplementedError

    sorted_result = sorted(zip(frags_smiles, frags_atom_idx), key=lambda x: len(x[1]))
    frags_smiles, frags_atom_idx = zip(*sorted_result)
    return frags_smiles, frags_atom_idx


def find_complete_seg(current_idx_set, current_match_list, all_atom_idx, num_element):
    if len(all_atom_idx) == 0:
        if len(current_idx_set) == num_element:
            return current_match_list
        else:
            return None

    raw_matches = all_atom_idx[0]
    all_matches_subset = []
    # trim the matches list
    matches = []
    for match in raw_matches:
        if any([x in current_idx_set for x in match]):
            continue
        matches.append(match)

    for L in reversed(range(1, min(len(matches) + 1, num_element - len(current_idx_set) + 1))):
        for subset in itertools.combinations(matches, L):
            subset = list(itertools.chain(*subset))
            if len(subset) == len(set(subset)) and \
                    len(set(subset + list(current_idx_set))) == len(subset) + len(current_idx_set):
                all_matches_subset.append(subset)
    # print('current idx set: ', current_idx_set, 'all match subset: ', all_matches_subset)

    for match in all_matches_subset:
        valid = True
        for i in match:
            if i in current_idx_set:
                valid = False
                break
        if valid:
            next_idx_set = copy.deepcopy(current_idx_set)
            next_match_list = copy.deepcopy(current_match_list)
            for i in match:
                next_idx_set.add(i)
            next_match_list.append(match)

            match_list = find_complete_seg(next_idx_set, next_match_list, all_atom_idx[1:], num_element)
            if match_list is not None:
                return match_list


def compute_pocket_frag_distance(pocket_centers, frag_centroid):
    all_distances = []
    for center in pocket_centers:
        distance = np.linalg.norm(frag_centroid - center, ord=2)
        all_distances.append(distance)
    return np.mean(all_distances)


def is_terminal_frag(mol, frag_atom_idx):
    split_bond_idx = []
    for bond_idx, bond in enumerate(mol.GetBonds()):
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if (start in frag_atom_idx) != (end in frag_atom_idx):
            split_bond_idx.append(bond_idx)
    return len(split_bond_idx) <= 1  # equals 0 when only one fragment is detected


def get_submol(mol, split_bond_idx, pocket_atom_idx):
    if len(pocket_atom_idx) == 0:
        return None
    elif len(pocket_atom_idx) == mol.GetNumAtoms() and len(split_bond_idx) == 0:
        return copy.deepcopy(mol)
    else:
        r = Chem.FragmentOnBonds(mol, split_bond_idx)
        frags = Chem.GetMolFrags(r)
        frags_overlap_atoms = [len(set(pocket_atom_idx).intersection(set(frag))) for frag in frags]
        hit_idx = np.argmax(frags_overlap_atoms)
        submol = Chem.GetMolFrags(r, asMols=True)[hit_idx]
        return submol


def extract_submols(mol, pocket_list, debug=False, verbose=False):
    # decompose molecules into fragments
    try:
        union_frags_smiles, possible_frags_atom_idx = decompose_molecule(mol)
    except:
        raise DecomposeError
    # each element is a group of fragments with the same type
    match_frags_list = find_complete_seg(set(), [], possible_frags_atom_idx, mol.GetNumAtoms())
    if match_frags_list is None:
        raise DecomposeError
    # flatten the matching list
    frags_smiles_list, frags_atom_idx_list = [], []
    for smiles, group_atom_idx in zip(union_frags_smiles, match_frags_list):
        query_frag_mol = Chem.MolFromSmiles(smiles)
        if len(group_atom_idx) == query_frag_mol.GetNumAtoms():
            frags_smiles_list.append(smiles)
            frags_atom_idx_list.append(group_atom_idx)
        else:
            assert len(group_atom_idx) % query_frag_mol.GetNumAtoms() == 0
            n_atoms = 0
            for match in mol.GetSubstructMatches(query_frag_mol):
                if all([atom_idx in group_atom_idx for atom_idx in match]):
                    frags_smiles_list.append(smiles)
                    frags_atom_idx_list.append([atom_idx for atom_idx in match])
                    n_atoms += len(match)
            assert n_atoms == len(group_atom_idx)
    # find centroid of each fragment
    ligand_pos = mol.GetConformer().GetPositions()
    dist_mat = np.zeros([len(frags_smiles_list), len(pocket_list)])

    all_frag_centroid = []
    for frag_idx, (frag_smiles, frag_atom_idx) in enumerate(zip(frags_smiles_list, frags_atom_idx_list)):
        frag_pos = np.array([ligand_pos[atom_idx] for atom_idx in frag_atom_idx])
        frag_centroid = np.mean(frag_pos, 0)
        all_frag_centroid.append(frag_centroid)

        for pocket_idx, pocket in enumerate(pocket_list):
            centers = [a.centroid for a in pocket.alphas]
            distance = compute_pocket_frag_distance(centers, frag_centroid)
            dist_mat[frag_idx, pocket_idx] = distance
    all_frag_centroid = np.array(all_frag_centroid)

    # clustering
    # number of clustering centers: number of pockets (arms) + 1 (scaffold)
    # 1. determine clustering centers
    terminal_mask = np.array([is_terminal_frag(mol, v) for v in frags_atom_idx_list])
    t_frag_idx = (terminal_mask == 1).nonzero()[0]
    nt_frag_idx = (terminal_mask == 0).nonzero()[0]

    pocket_idx, frag_idx = linear_sum_assignment(dist_mat[t_frag_idx].T)
    arms_frag_idx = np.array([t_frag_idx[idx] for idx in frag_idx])
    clustering_centers = [all_frag_centroid[idx] for idx in arms_frag_idx]
    # linear_sum_assignment will handle the case where the amount of arms is greater than the number of pockets
    # if the number of arms is less than the number of pockets, supplement centroid of pocket's alpha atoms
    cluster_pocket_idx = list(pocket_idx)
    if len(clustering_centers) < len(pocket_list):
        if verbose:
            print('warning: less arms than pockets')
        add_pocket_idx = list(set(range(len(pocket_list))) - set(pocket_idx))
        for p_idx in add_pocket_idx:
            centers = [a.centroid for a in pocket_list[p_idx].alphas]
            pocket_centroid = np.mean(centers, 0)
            clustering_centers.append(pocket_centroid)
            cluster_pocket_idx.append(p_idx)
    assert len(clustering_centers) == len(pocket_list)

    # select the frag centroid which is farthest to all existing centers as the scaffold clustering center
    # it is possible that only arm fragments are detected
    non_arm_frag_idx = np.array([idx for idx in range(len(all_frag_centroid)) if idx not in arms_frag_idx])
    if len(non_arm_frag_idx) > 0:
        scaffold_frag_idx = non_arm_frag_idx[
            np.argmax(distance_matrix(all_frag_centroid[non_arm_frag_idx], clustering_centers).sum(-1))]
        clustering_centers.append(all_frag_centroid[scaffold_frag_idx])
    else:
        scaffold_frag_idx = np.array([], dtype=np.int)

    if debug:
        print(f't frag idx: {t_frag_idx} nt frag idx: {nt_frag_idx} '
              f'arms frag idx: {arms_frag_idx} non arm frag idx: {non_arm_frag_idx}')

    # 2. determine assignment
    # todo: can be improved, like updating clustering center
    frag_cluster_dist_mat = distance_matrix(all_frag_centroid, clustering_centers)
    assignment = -1 * np.ones(len(all_frag_centroid)).astype(np.int64)
    assignment[arms_frag_idx] = pocket_idx

    # todo: order problem
    assignment[scaffold_frag_idx] = len(clustering_centers) - 1
    for idx in range(len(all_frag_centroid)):
        assign_cluster_idx = frag_cluster_dist_mat[idx].argmin()
        if assign_cluster_idx == len(clustering_centers) - 1:
            # directly assign scaffold
            assignment[idx] = len(clustering_centers) - 1
        else:
            assign_pocket_idx = cluster_pocket_idx[assign_cluster_idx]
            # arms --> check validity
            current_atom_idx = []
            for assign_frag_idx in (assignment == assign_pocket_idx).nonzero()[0]:
                current_atom_idx += frags_atom_idx_list[assign_frag_idx]
            current_atom_idx += frags_atom_idx_list[idx]

            if is_terminal_frag(mol, current_atom_idx):
                assignment[idx] = assign_pocket_idx
            else:
                assignment[idx] = len(clustering_centers) - 1

    # 3. construct submols given assignment
    all_submols = []
    scaffold_bond_idx = []
    all_arm_atom_idx = []
    valid_pocket_id = []
    for pocket_id in range(len(pocket_list)):
        arm_atom_idx = []
        for assigned_idx in (assignment == pocket_id).nonzero()[0]:
            arm_atom_idx += frags_atom_idx_list[assigned_idx]

        split_bond_idx = []
        for bond_idx, bond in enumerate(mol.GetBonds()):
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if (start in arm_atom_idx) != (end in arm_atom_idx):
                split_bond_idx.append(bond_idx)
        assert len(split_bond_idx) <= 1, split_bond_idx
        scaffold_bond_idx += split_bond_idx

        match_submol = get_submol(mol, split_bond_idx, arm_atom_idx)
        if len(arm_atom_idx) > 0:
            valid_pocket_id.append(pocket_id)
            assert match_submol is not None
            all_arm_atom_idx.append(arm_atom_idx)
            all_submols.append(match_submol)

    scaffold_atom_idx = []
    for assigned_idx in (assignment == len(pocket_list)).nonzero()[0]:
        scaffold_atom_idx += frags_atom_idx_list[assigned_idx]

    scaffold_submol = get_submol(mol, scaffold_bond_idx, scaffold_atom_idx)
    all_submols.append(scaffold_submol)
    flat_arm_atom_idx = list(itertools.chain(*all_arm_atom_idx))
    assert len(flat_arm_atom_idx + scaffold_atom_idx) == len(set(flat_arm_atom_idx + scaffold_atom_idx))
    assert set(flat_arm_atom_idx + scaffold_atom_idx) == set(range(mol.GetNumAtoms()))
    all_submol_atom_idx = all_arm_atom_idx + [scaffold_atom_idx]
    return all_frag_centroid, assignment, all_submol_atom_idx, all_submols, valid_pocket_id


def extract_subpockets(protein, pocket, method, **kwargs):
    if method == 'v1':
        # Method 1: union of lining atom idx / lining residue idx
        pocket_lining_atoms = [atom for atom in kwargs['mdtraj_protein'].atom_slice(pocket.lining_atoms_idx).top.atoms]
        pocket_atom_serial = [atom.serial for atom in pocket_lining_atoms]
        # pocket_res_idx = [atom.residue.resSeq for atom in pocket_lining_atoms]

        selected_atom_serial, selected_residues = [], []
        sel_idx = set()
        for atom in protein.atoms:
            if atom['atom_id'] in pocket_atom_serial:
                chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
                sel_idx.add(chain_res_id)

        for res in protein.residues:
            if res['chain_res_id'] in sel_idx:
                selected_residues.append(res)
                selected_atom_serial += [protein.atoms[a_idx]['atom_id'] for a_idx in res['atoms']]

    elif method == 'v2':
        # Method 2: alpha atom --> sphere with a large radius
        centers = [a.centroid for a in pocket.alphas]
        selected_atom_serial, selected_residues = protein.query_residues_centers(
            centers, radius=kwargs['protein_radius'])

    elif method == 'v3':
        # Method 3: atom-level query but select the whole residue
        centers = [a.centroid for a in pocket.alphas]
        selected_atom_serial, selected_residues = protein.query_residues_atom_centers(
            centers, radius=kwargs['protein_radius'])

    elif method == 'submol_radius':
        centers = kwargs['submol'].GetConformer(0).GetPositions()
        selected_atom_serial, selected_residues = protein.query_residues_centers(
            centers, radius=kwargs['protein_radius'])

    else:
        raise NotImplementedError

    return selected_atom_serial, selected_residues


def union_pocket_residues(all_pocket_residues):
    selected = []
    sel_idx = set()
    for pocket_r in all_pocket_residues:
        for r in pocket_r:
            if r['chain_res_id'] not in sel_idx:
                selected.append(r)
                sel_idx.add(r['chain_res_id'])
    return selected


def mark_in_range(query_points, ref_points, cutoff=1.6):
    indices = np.where(distance_matrix(query_points, ref_points) <= cutoff)[0]
    indices = np.unique(indices)
    query_bool = np.zeros(len(query_points), dtype=bool)
    query_bool[indices] = 1
    return query_bool

# def query_ligand_centers(ligand, centers, radius=4):
#     selected = []
#     sel_idx = set()
#     for center in centers:
#         for i, pos in enumerate(ligand['pos']):
#             distance = np.linalg.norm(pos - center, ord=2)
#             if distance < radius and i not in sel_idx:
#                 selected.append(pos)
#                 sel_idx.add(i)
#     return np.array(selected), list(sel_idx)
