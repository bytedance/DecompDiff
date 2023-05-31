# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import numpy as np
from utils.preprocess import mark_in_range
from mdtraj.core import element
from mdtraj.core.trajectory import Trajectory


def compute_occupancy(pocket, ligand_pos):
    centers = [a.centroid for a in pocket.alphas]
    contact_mask = mark_in_range(centers, ligand_pos)
    occupied_space = np.sum(np.array([alpha.nonpolar_space for alpha in pocket.alphas]) * contact_mask)
    occupancy = occupied_space / pocket.nonpolar_space
    return occupancy


def compute_polar_ratio(receptor: Trajectory, pocket, rdmol):
    pocket_lining_atoms = [atom for atom in receptor.atom_slice(pocket.lining_atoms_idx).top.atoms]
    pocket_is_polar = np.array(
        [atom.element in [element.oxygen, element.nitrogen, element.sulfur] for atom in pocket_lining_atoms]).astype(
        int)
    pocket_polar_ratio = np.sum(pocket_is_polar) / len(pocket_lining_atoms)

    ligand_atom_type = [atom.GetAtomicNum() for atom in rdmol.GetAtoms() if atom.GetAtomicNum() != 0]
    ligand_is_polar = np.array([atom_type in [7, 8, 16] for atom_type in ligand_atom_type]).astype(int)
    ligand_polar_ratio = np.sum(ligand_is_polar) / len(ligand_atom_type)
    return pocket_polar_ratio, ligand_polar_ratio
