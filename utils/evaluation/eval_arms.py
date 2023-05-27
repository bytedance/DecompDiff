# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
