# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import itertools
import os
import pickle
from collections import defaultdict

import lmdb
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

from utils.data import PDBProtein
from utils.data import ProteinLigandData, torchify_dict, parse_sdf_file
from utils.prior import compute_golden_prior_from_data


def get_decomp_dataset(config, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = DecompPLPairDataset(root, mode=config.mode,
                                      include_dummy_atoms=config.include_dummy_atoms, version=config.version, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split_by_name = torch.load(config.split)
        split = {
            k: [dataset.name2id[n[1][:-4]] for n in names if n[1][:-4] in dataset.name2id]
            for k, names in split_by_name.items()
        }
        for k, v in split.items():
            split[k] = list(itertools.chain(*v))
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset


class DecompPLPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, mode='full',
                 include_dummy_atoms=False, kekulize=True, version='v1'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.mode = mode  # ['arms', 'scaffold', 'full']
        self.include_dummy_atoms = include_dummy_atoms
        self.kekulize = kekulize
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_{mode}_{version}.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path),
                                         os.path.basename(self.raw_path) + f'_{mode}_{version}_name2id.pt')

        self.transform = transform
        self.mode = mode
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        print('Load dataset from %s' % self.processed_path)
        if not os.path.exists(self.name2id_path):
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _precompute_name2id(self):
        name2id = defaultdict(list)
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = data.src_ligand_filename[:-4]
            # name = (data.src_protein_filename, data.src_ligand_filename)
            name2id[name].append(i)
        torch.save(name2id, self.name2id_path)

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        num_data = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, meta_info in enumerate(tqdm(index)):
                try:
                    with open(meta_info['data']['meta_file'], 'rb') as f:
                        m = pickle.load(f)['data']
                    num_arms, num_scaffold = m['num_arms'], m['num_scaffold']
                    if self.mode == 'full':
                        protein = PDBProtein(m['protein_file'])
                        protein_dict = protein.to_dict_atom()
                        ligand_dict = parse_sdf_file(m['ligand_file'], kekulize=self.kekulize)
                        num_protein_atoms, num_ligand_atoms = len(protein.atoms), ligand_dict['rdmol'].GetNumAtoms()
                        assert num_ligand_atoms == sum([len(x) for x in m['all_submol_atom_idx']])

                        # extract pocket atom mask
                        protein_atom_serial = [atom['atom_id'] for atom in protein.atoms]
                        pocket_atom_masks = []
                        assert len(m['all_pocket_atom_serial']) == num_arms
                        for pocket_atom_serial in m['all_pocket_atom_serial']:
                            pocket_atom_idx = [protein_atom_serial.index(i) for i in pocket_atom_serial]
                            pocket_atom_mask = torch.zeros(num_protein_atoms, dtype=torch.bool)
                            pocket_atom_mask[pocket_atom_idx] = 1
                            pocket_atom_masks.append(pocket_atom_mask)
                        pocket_atom_masks = torch.stack(pocket_atom_masks)

                        # extract ligand atom mask
                        ligand_atom_mask = torch.zeros(num_ligand_atoms, dtype=int)
                        for arm_idx, atom_idx in enumerate(m['all_submol_atom_idx']):
                            if arm_idx == len(m['all_submol_atom_idx']) - 1:
                                ligand_atom_mask[atom_idx] = -1
                            else:
                                ligand_atom_mask[atom_idx] = arm_idx
                        assert len(ligand_atom_mask.unique()) == num_arms + num_scaffold

                        data = ProteinLigandData.from_protein_ligand_dicts(
                            protein_dict=torchify_dict(protein_dict),
                            ligand_dict=torchify_dict(ligand_dict),
                        )
                        data.src_protein_filename = meta_info['src_protein_filename']
                        data.src_ligand_filename = meta_info['src_ligand_filename']
                        for k, v in meta_info['data'].items():
                            data[k] = v
                        data.num_arms, data.num_scaffold = num_arms, num_scaffold
                        data.pocket_atom_masks, data.ligand_atom_mask = pocket_atom_masks, ligand_atom_mask
                        data = compute_golden_prior_from_data(data)
                        data = data.to_dict()  # avoid torch_geometric version issue
                        txn.put(
                            # key=str(num_data).encode(),
                            key=f'{num_data:08d}'.encode(),
                            value=pickle.dumps(data)
                        )
                        num_data += 1

                    elif self.mode == 'arms':
                        frags_sdf_path = m['sub_ligand_file']
                        frags = list(Chem.SDMolSupplier(frags_sdf_path))
                        for arm_idx in range(num_arms):
                            protein = PDBProtein(m['sub_pocket_files'][arm_idx])
                            protein_dict = protein.to_dict_atom()
                            if self.include_dummy_atoms:
                                ligand_dict = parse_sdf_file(frags[arm_idx])
                            else:
                                du = Chem.MolFromSmiles('*')
                                nodummy_frag = AllChem.ReplaceSubstructs(
                                    frags[arm_idx], du, Chem.MolFromSmiles('[H]'), True)[0]
                                nodummy_frag = Chem.RemoveHs(nodummy_frag)
                                ligand_dict = parse_sdf_file(nodummy_frag)

                            data = ProteinLigandData.from_protein_ligand_dicts(
                                protein_dict=torchify_dict(protein_dict),
                                ligand_dict=torchify_dict(ligand_dict),
                            )
                            if data.protein_pos.size(0) == 0:
                                continue
                            data.src_protein_filename = meta_info['src_protein_filename']
                            data.src_ligand_filename = meta_info['src_ligand_filename']
                            data.arm_idx = arm_idx
                            data.occupancy = m['pocket_occupancies_by_submol'][arm_idx]
                            for k, v in meta_info['data'].items():
                                data[k] = v
                            data.num_arms, data.num_scaffold = num_arms, num_scaffold
                            # data.pocket_atom_masks, data.ligand_atom_mask = pocket_atom_masks, ligand_atom_mask
                            data = data.to_dict()  # avoid torch_geometric version issue
                            txn.put(
                                key=f'{num_data:08d}'.encode(),
                                value=pickle.dumps(data)
                            )
                            num_data += 1

                    elif self.mode == 'scaffold':
                        raise NotImplementedError

                    else:
                        raise ValueError
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, meta_info['src_ligand_filename'], ))
                    continue
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        # assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_raw_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        return data
        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--mode', type=str, default='full')
    parser.add_argument('--dummy', type=eval, default=False)
    parser.add_argument('--keku', type=eval, default=True)
    parser.add_argument('--version', type=str, required=True)
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    dataset = DecompPLPairDataset(args.path, mode=args.mode,
                                  include_dummy_atoms=args.dummy, kekulize=args.keku, version=args.version)
    print(len(dataset), dataset[0])
