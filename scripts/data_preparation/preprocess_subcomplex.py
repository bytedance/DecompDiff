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


import argparse
import multiprocessing as mp
import os.path
import pickle
from functools import partial

import alphaspace2 as al
import mdtraj
from rdkit import Chem

from utils.data import convert_sdf_to_pdb, PDBProtein
from utils.misc import *
from utils.preprocess import extract_subpockets, extract_submols, union_pocket_residues, mark_in_range

KMAP = {'Ki': 1, 'Kd': 2, 'IC50': 3}


def parse_pdbbind_index_file(raw_path, subset='refined'):
    all_index = []
    version = int(raw_path.rstrip('/')[-4:])
    assert version >= 2016
    if subset == 'refined':
        data_path = os.path.join(raw_path, f'refined-set')
        index_path = os.path.join(data_path, 'index', f'INDEX_refined_data.{version}')
    elif subset == 'general':
        data_path = os.path.join(raw_path, f'general-set-except-refined')
        index_path = os.path.join(data_path, 'index', f'INDEX_general_PL_data.{version}')
    else:
        raise ValueError(subset)

    all_files = os.listdir(data_path)
    with open(index_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        index, res, year, pka, kv = line.split('//')[0].strip().split()
        kind = [v for k, v in KMAP.items() if k in kv]
        assert len(kind) == 1
        if index in all_files:
            all_index.append({
                'pdb_index': index,
                'src_protein_filename': os.path.join(index, f'{index}_protein.pdb'),
                'src_ligand_filename': os.path.join(index, f'{index}_ligand.sdf'),
                'resolution': res,
                'pka': pka,
                'pkd_type': kind[0],
            })

    return sorted(all_index, key=lambda x: x['pdb_index'])


def parse_crossdocked_index_file(split_fn):
    all_index = []
    split_index = torch.load(split_fn)
    for k, v in split_index.items():
        for protein_fn, ligand_fn in v:
            protein_fn = os.path.join(os.path.dirname(protein_fn),
                                      os.path.basename(protein_fn)[:10] + '.pdb')
            all_index.append({
                'src_protein_filename': protein_fn,
                'src_ligand_filename': ligand_fn
            })
    return all_index


# @profile
def extract_subcomplex(config, protein_path, ligand_sdf_path,
                       save_snapshot_path=None, save_path=None, verbose=False):

    tmp_ligand_pdb_path = ligand_sdf_path.replace('.sdf', '.pdb')
    tmp_ligand_pdb_path = os.path.join('./tmp', os.path.basename(tmp_ligand_pdb_path))
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')

    try:
        # Convert sdf format ligand file to pdb format
        convert_sdf_to_pdb(ligand_sdf_path, tmp_ligand_pdb_path)
        # Load in receptor and ligand seperately using mdtraj
        receptor = mdtraj.load(protein_path)
        binder = mdtraj.load(tmp_ligand_pdb_path)
        os.remove(tmp_ligand_pdb_path)
        # Initialize a snapshot object, this will contain the receptor and the binder informations
        ss = al.Snapshot()
        for k, v in config.alphaspace.items():
            setattr(ss, k, v)
        # Run the snapshot object by feeding it receptor and binder mdtraj objects.
        ss.run(receptor=receptor, binder=binder)
    except:
        raise AlphaSpaceError

    pocket_list = [p for p in sorted(ss.pockets, key=lambda i: i.nonpolar_space, reverse=True) if p.isContact]
    if len(pocket_list) == 0:
        raise ExtractPocketError

    all_pocket_occupancy = []
    for p_idx, p in enumerate(pocket_list):
        all_pocket_occupancy.append(p.occupancy_nonpolar)
        if verbose:
            print(
                "Pocket {} has alpha-space of {} A3, BetaScore of {:.1f} kcal/mol and is {:.0f}% occupied".format(
                    p_idx, round(p.nonpolar_space), p.score, (p.occupancy_nonpolar * 100)
                )
            )
        # print(p.occupiedNonpolarSpace, p.nonpolar_space)
    if save_snapshot_path:
        al.write_snapshot(folder_path=save_snapshot_path, snapshot=ss, receptor=receptor, binder=binder)

    # ligand = parse_sdf_file(ligand_sdf_path)
    rdmol = next(iter(Chem.SDMolSupplier(ligand_sdf_path, removeHs=True, sanitize=True)))
    if rdmol is None:
        ligand_mol2_path = ligand_sdf_path.replace('.sdf', '.mol2')
        rdmol = Chem.MolFromMol2File(ligand_mol2_path)
    if rdmol is None:
        raise SDFParsingError

    # Extract sub-molecules
    all_frag_centroid, assignment, all_submol_atom_idx, all_submols, valid_pocket_id = \
        extract_submols(rdmol, pocket_list)

    # Extract pocket protein atoms / residues
    all_pocket_atom_serial, all_pocket_residues = [], []
    all_pocket_occupancy_by_submol = []
    protein = PDBProtein(protein_path)
    for i, pocket_id in enumerate(valid_pocket_id):
        pocket = pocket_list[pocket_id]
        selected_atom_serial, selected_residues = extract_subpockets(
            protein, pocket, method=config.pocket.protein_ext_method,
            submol=all_submols[i],
            mdtraj_protein=receptor, protein_radius=config.pocket.protein_radius,
        )

        all_pocket_atom_serial.append(selected_atom_serial)
        all_pocket_residues.append(selected_residues)

        centers = [a.centroid for a in pocket.alphas]
        contact_mask = mark_in_range(centers, all_submols[i].GetConformer(0).GetPositions())
        occupied_space = np.sum(np.array([alpha.nonpolar_space for alpha in pocket.alphas]) * contact_mask)
        occupancy = occupied_space / pocket.nonpolar_space
        all_pocket_occupancy_by_submol.append(occupancy)

    sub_pocket_dest_files, sub_ligand_dest = [], None
    ori_protein_fn = os.path.basename(protein_path)
    ori_ligand_fn = os.path.basename(ligand_sdf_path)
    union_pocket_dest, union_ligand_dest = None, None

    if save_path:
        if config.dataset.name == 'pdbbind':
            union_pocket_dest = os.path.join(save_path, ori_protein_fn)[:-4] + '_pocket.pdb'
        elif config.dataset.name == 'crossdocked':
            union_pocket_dest = os.path.join(save_path, ori_ligand_fn)[:-4] + '_pocket.pdb'
        else:
            raise ValueError
        union_ligand_dest = os.path.join(save_path, ori_ligand_fn)

        # write overall pocket and molecule
        os.makedirs(save_path, exist_ok=True)
        # shutil.copyfile(
        #     src=ligand_sdf_path,
        #     dst=union_ligand_dest
        # )
        w = Chem.SDWriter(union_ligand_dest)
        w.write(rdmol)
        w.close()
        if config.pocket.protein_ext_method == 'submol_radius':
            _, union_residues = protein.query_residues_centers(rdmol.GetConformer(0).GetPositions(),
                                                               config.pocket.protein_radius)
        else:
            union_residues = union_pocket_residues(all_pocket_residues)
        pdb_block_pocket = protein.residues_to_pdb_block(union_residues)
        with open(union_pocket_dest, 'w') as f:
            f.write(pdb_block_pocket)

        # write sub-pocket
        for idx, pocket_residues in enumerate(all_pocket_residues):
            pocket_dest = union_pocket_dest[:-4] + '_%d.pdb' % idx
            # pocket_dest = os.path.join(save_path, pocket_fn)
            pdb_block_pocket = protein.residues_to_pdb_block(pocket_residues)
            with open(pocket_dest, 'w') as f:
                f.write(pdb_block_pocket)
            sub_pocket_dest_files.append(pocket_dest)

        # write sub-molecule
        ligand_fn = ori_ligand_fn[:-4] + '_frags.sdf'
        sub_ligand_dest = os.path.join(save_path, ligand_fn)
        w = Chem.SDWriter(sub_ligand_dest)
        for m in all_submols:
            if m is not None:
                w.write(m)
        w.close()

    return {
        'al_snapshot': ss,
        'all_pockets': [pocket_list[pocket_id] for pocket_id in valid_pocket_id],
        'all_pocket_atom_serial': all_pocket_atom_serial,
        'all_pocket_residues': all_pocket_residues,
        'all_submols': all_submols,
        'all_submol_atom_idx': all_submol_atom_idx,
        'protein_file': union_pocket_dest,
        'ligand_file': union_ligand_dest,
        'sub_pocket_files': sub_pocket_dest_files,
        'sub_ligand_file': sub_ligand_dest,

        'num_pockets': len(pocket_list),
        'num_frags': len(all_frag_centroid),
        'num_arms': len(valid_pocket_id),
        'num_scaffold': 1 if all_submols[-1] is not None else 0,
        'pocket_occupancies_by_mol': all_pocket_occupancy,
        'pocket_occupancies_by_submol': all_pocket_occupancy_by_submol
    }


# @profile
def process_item(item, config):
    if config.dataset.name == 'pdbbind':
        # pdb_idx, res, year, pka, kind = item
        pdb_idx = item['pdb_index']
        if config.dataset.split == 'refined':
            pdb_path = os.path.join(config.dataset.path, 'refined-set', pdb_idx)
        elif config.dataset.split == 'general':
            pdb_path = os.path.join(config.dataset.path, 'general-set-except-refined', pdb_idx)
        else:
            raise ValueError(args.subset)
        protein_path = os.path.join(pdb_path, f'{pdb_idx}_protein.pdb')
        ligand_path = os.path.join(pdb_path, f'{pdb_idx}_ligand.sdf')
        save_path = os.path.join(args.dest, pdb_idx)

    elif config.dataset.name == 'crossdocked':
        protein_path = os.path.join(config.dataset.path, item['src_protein_filename'])
        ligand_path = os.path.join(config.dataset.path, item['src_ligand_filename'])
        save_path = os.path.join(args.dest, os.path.dirname(item['src_ligand_filename']))
        pdb_idx = '%s + %s' % (os.path.basename(protein_path), os.path.basename(ligand_path))

    else:
        raise NotImplementedError

    try:
        # print(f'begin process PDB {pdb_idx}')
        r = extract_subcomplex(config, protein_path, ligand_path, save_path=save_path)
        print(f"PDB {pdb_idx}: num_pockets: {r['num_pockets']} num_frags: {r['num_frags']} "
              f"num_arms: {r['num_arms']} num_scaffold: {r['num_scaffold']}")
    except AlphaSpaceError:
        print(f"{pdb_idx}: AlphaSpace run fail!")
        r = 'al_fail'
    except SDFParsingError:
        print(f"{pdb_idx}: Parse sdf file fail!")
        r = 'sdf_fail'
    except DecomposeError:
        print(f"{pdb_idx}: Extract sub-molecule fail!")
        r = 'submol_fail'
    except ExtractPocketError:
        print(f"{pdb_idx}: Extract sub-pocket fail!")
        r = 'subpok_fail'
    except OSError as e:
        print(e)
        r = 'os_fail'
    except Exception:
        print(f"{pdb_idx}: Other fail!")
        r = 'other_fail'

    result_dict = {
        **item,
        'data': r
    }

    if isinstance(r, dict):
        # success
        prefix = os.path.basename(item['src_ligand_filename'])[:-4]
        meta_fn = os.path.join(save_path, prefix + '_meta.pkl')
        with open(meta_fn, 'wb') as f:
            pickle.dump(result_dict, f)

        result_index = {
            **item,
            'data': {
                'protein_file': r['protein_file'],
                'ligand_file': r['ligand_file'],
                'sub_pocket_files': r['sub_pocket_files'],
                'sub_ligand_file': r['sub_ligand_file'],
                'meta_file': meta_fn
            }
        }

    else:
        result_index = result_dict

    return result_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--dest', type=str, default='./data/crossdocked_v1.1_rmsd1.0_processed')
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.dataset.name == 'pdbbind':
        all_index = parse_pdbbind_index_file(config.dataset.path, config.dataset.split)
    elif config.dataset.name == 'crossdocked':
        all_index = parse_crossdocked_index_file(config.dataset.split)
    else:
        raise NotImplementedError

    results = []
    with mp.Pool(args.num_workers) as pool:
        multiple_fns = [pool.apply_async(partial(process_item, config=config), (index,)) for index in all_index]
        for fn, index in tqdm(zip(multiple_fns, all_index), total=len(all_index)):
            try:
                result = fn.get(timeout=60)
            except mp.TimeoutError:
                results.append({
                    **index,
                    'data': 'timeout_fail'
                })
                continue
            results.append(result)

    valid_index_pocket = []
    al_fail, sdf_parsing_fail, timeout_fail, submol_fail, subpok_fail, os_fail, other_fail = [], [], [], [], [], [], []
    for index in results:
        protein_fn, ligand_fn, r = index['src_protein_filename'], index['src_ligand_filename'], index['data']
        if r == 'al_fail':
            al_fail.append((protein_fn, ligand_fn))
        elif r == 'sdf_fail':
            sdf_parsing_fail.append((protein_fn, ligand_fn))
        elif r == 'timeout_fail':
            timeout_fail.append((protein_fn, ligand_fn))
        elif r == 'submol_fail':
            submol_fail.append((protein_fn, ligand_fn))
        elif r == 'subpok_fail':
            subpok_fail.append((protein_fn, ligand_fn))
        elif r == 'os_fail':
            os_fail.append((protein_fn, ligand_fn))
        elif r == 'other_fail':
            other_fail.append((protein_fn, ligand_fn))
        elif isinstance(r, dict):
            valid_index_pocket.append(index)
        else:
            raise ValueError

    # merge refined set if necessary
    # if args.subset == 'general' and args.refined_index_pkl is not None:
    #     with open(args.refined_index_pkl, 'rb') as f:
    #         refined_index = pickle.load(f)
    #     valid_index_pocket += refined_index

    # save index
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    with open(os.path.join(args.dest, 'index.pkl'), 'wb') as f:
        pickle.dump(valid_index_pocket, f)

    all_fail_index = {
        'al_fail': al_fail,
        'sdf_parsing_fail': sdf_parsing_fail,
        'timeout_fail': timeout_fail,
        'submol_fail': submol_fail,
        'subpok_fail': subpok_fail,
        'os_fail': os_fail,
        'other_fail': other_fail
    }

    with open(os.path.join(args.dest, 'fail_index.pkl'), 'wb') as f:
        pickle.dump(all_fail_index, f)

    print('Done. %d protein-ligand pairs in total.' % len(valid_index_pocket))
    print(
        f'AlphaSpace fail: %d SDF parsing fail: %d SubMol extraction fail: %d SubPocket extraction fail: %d '
        f'Timeout fail: %d OS fail: %d Other fail: %d' % (
            len(al_fail), len(sdf_parsing_fail), len(submol_fail), len(subpok_fail),
            len(timeout_fail), len(os_fail), len(other_fail)
        )
    )
