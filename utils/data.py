# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from openbabel import openbabel as ob
import torch


ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}


def convert_sdf_to_pdb(sdf_path, pdb_path):
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdb")

    mol = ob.OBMol()
    obConversion.ReadFile(mol, sdf_path)

    # mol.AddHydrogens()

    # print mol.NumAtoms()
    # print mol.NumBonds()
    # print mol.NumResidues()

    obConversion.WriteFile(mol, pdb_path)


class PDBProtein(object):
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            atom = self.atoms[residue['atoms'][0]]
            residue['chain_res_id'] = '%s_%s_%d_%s' % (
                atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            residue['center_of_mass'] = sum_pos / sum_mass

        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.long),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=np.bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.long)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.long),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_centers(self, centers, radius, criterion='center_of_mass'):
        selected_atom_serial = []
        selected_residues = []
        sel_idx = set()
        for center in centers:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected_residues.append(residue)
                    sel_idx.add(i)

        for res in selected_residues:
            selected_atom_serial += [self.atoms[a_idx]['atom_id'] for a_idx in res['atoms']]
        return selected_atom_serial, selected_residues

    def query_residues_atom_centers(self, centers, radius):
        selected_atom_serial = []
        selected_residues = []
        sel_idx = set()
        for center in centers:
            for i, atom in enumerate(self.atoms):
                atom_pos = np.array([atom['x'], atom['y'], atom['z']])
                distance = np.linalg.norm(atom_pos - center, ord=2)
                chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
                if distance < radius and chain_res_id not in sel_idx:
                    sel_idx.add(chain_res_id)

        for res in self.residues:
            if res['chain_res_id'] in sel_idx:
                selected_residues.append(res)
                selected_atom_serial += [self.atoms[a_idx]['atom_id'] for a_idx in res['atoms']]
        return selected_atom_serial, selected_residues

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            # print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        centers = ligand['pos'] if isinstance(ligand, dict) else ligand  # ligand pos
        for center in centers:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block


def parse_pdbbind_index_file(path):
    pdb_id = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        pdb_id.append(line.split()[0])
    return pdb_id


def change_formal_charge(mol):
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)

    return mol


def process_from_mol(rdmol):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    # Get hybridization in the order of atom idx.
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    ptable = Chem.GetPeriodicTable()

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int)

    # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.long)
    edge_type = np.array(edge_type, dtype=np.long)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'rdmol': rdmol,
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization
    }
    return data


def parse_sdf_file(path, kekulize=True):
    # Remove Hydrogens.
    if isinstance(path, str):
        rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True, sanitize=True)))
    elif isinstance(path, Chem.Mol):
        rdmol = path
    else:
        raise ValueError(type(path))
    # rdmol = change_formal_charge(rdmol)
    if kekulize:
        Chem.Kekulize(rdmol)

    data = process_from_mol(rdmol)
    data['smiles'] = Chem.MolToSmiles(rdmol, kekuleSmiles=kekulize)
    return data

###
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

FOLLOW_BATCH = []  # ['protein_element', 'ligand_context_element', 'pos_real', 'pos_fake']


class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

            instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if
                                                      instance.ligand_bond_index[0, k].item() == i] for i in
                                           instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        elif key == 'ligand_context_bond_index':
            return self['ligand_context_element'].size(0)

        elif key == 'mask_ctx_edge_index_0':
            return self['ligand_masked_element'].size(0)
        elif key == 'mask_ctx_edge_index_1':
            return self['ligand_context_element'].size(0)
        elif key == 'mask_compose_edge_index_0':
            return self['ligand_masked_element'].size(0)
        elif key == 'mask_compose_edge_index_1':
            return self['compose_pos'].size(0)

        elif key == 'compose_knn_edge_index':  # edges for message passing of encoder
            return self['compose_pos'].size(0)

        elif key == 'real_ctx_edge_index_0':
            return self['pos_real'].size(0)
        elif key == 'real_ctx_edge_index_1':
            return self['ligand_context_element'].size(0)
        elif key == 'real_compose_edge_index_0':  # edges for edge type prediction
            return self['pos_real'].size(0)
        elif key == 'real_compose_edge_index_1':
            return self['compose_pos'].size(0)

        elif key == 'real_compose_knn_edge_index_0':  # edges for message passing of  field
            return self['pos_real'].size(0)
        elif key == 'fake_compose_knn_edge_index_0':
            return self['pos_fake'].size(0)
        elif (key == 'real_compose_knn_edge_index_1') or (key == 'fake_compose_knn_edge_index_1'):
            return self['compose_pos'].size(0)

        elif (key == 'idx_protein_in_compose') or (key == 'idx_ligand_ctx_in_compose'):
            return self['compose_pos'].size(0)

        elif key == 'index_real_cps_edge_for_atten':
            return self['real_compose_edge_index_0'].size(0)
        elif key == 'tri_edge_index':
            return self['compose_pos'].size(0)

        elif key == 'idx_generated_in_ligand_masked':
            return self['ligand_masked_element'].size(0)
        elif key == 'idx_focal_in_compose':
            return self['compose_pos'].size(0)
        elif key == 'idx_protein_all_mask':
            return self['compose_pos'].size(0)

        # for decomp
        elif key == 'ligand_decomp_mask':
            return self['num_arms'] + 1  # though there is no scaffold, decomp num atoms / centers still occupy 1 dim
        elif key == 'ligand_decomp_group_idx' or key == 'protein_decomp_group_idx':
            return self['max_decomp_group']
        elif key == 'ligand_fc_bond_index' or key == 'ligand_full_bond_index':
            return self['ligand_atom_mask'].size(0)  # only set ligand atom mask during sampling
        else:
            return super().__inc__(key, value)


class ProteinLigandDataLoader(DataLoader):

    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            follow_batch=['ligand_element', 'protein_element'],
            **kwargs
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)


def batch_from_data_list(data_list):
    return Batch.from_data_list(data_list, follow_batch=['ligand_element', 'protein_element'])


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


class ProteinLigandFragData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandFragData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if
                                                  instance.ligand_bond_index[0, k].item() == i] for i in
                                       instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        # print('key: ', key)
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        elif key == 'ligand_context_bond_index':
            return self['ligand_context_pos'].size(0)
        elif key == 'ligand_frag_edge_index':
            return self['ligand_frag_types'].size(0)
        elif key == 'ligand_torsion_4atoms':
            return self['ligand_context_pos'].size(0)

        elif key == 'idx_ligand_ctx_in_compose':
            return self['compose_pos'].size(0)
        elif key == 'idx_protein_in_compose':
            return self['compose_pos'].size(0)
        elif key == 'compose_knn_edge_index':
            return self['compose_pos'].size(0)

        elif key == 'idx_ligand_dummy_in_compose':
            return self['compose_pos'].size(0)
        elif key == 'frontier_idx_in_ctx':
            return self['compose_pos'].size(0)

        elif key == 'ligand_gen_bond_index':
            return self['ligand_gen_pos'].size(0)
        elif key == 'gen_connect_idx_in_ctx':
            return self['compose_pos'].size(0)
        elif key == 'idx_cand_dummy_in_gen':
            return self['ligand_gen_pos'].size(0)
        elif key == 'cand_dummy_batch':
            return self['cand_dummy_batch'].max() + 1

        elif key == 'idx_ligand_ctx_in_cg_compose':
            return self['cg_compose_pos'].size(0)
        elif key == 'idx_protein_in_cg_compose':
            return self['cg_compose_pos'].size(0)
        elif key == 'ligand_ctxgen_bond_index':
            return self['cg_compose_pos'].size(0)
        elif key == 'cg_compose_knn_edge_index':
            return self['cg_compose_pos'].size(0)
        elif key == 'idx_torsion_in_cg_compose':
            return self['cg_compose_pos'].size(0)

        else:
            return super().__inc__(key, value)
