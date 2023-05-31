# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import py3Dmol
import os
import copy

import torch
from rdkit import Chem
from rdkit.Chem import Draw
import pickle
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
# IPythonConsole.drawOptions.addAtomIndices = True
# IPythonConsole.molSize = 400,400


def visualize_complex(pdb_block, sdf_block, show_protein_surface=True, show_ligand=True, show_ligand_surface=True):
    view = py3Dmol.view()

    # Add protein to the canvas
    view.addModel(pdb_block, 'pdb')
    if show_protein_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
    else:
        view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0}})

    # Add ligand to the canvas
    if show_ligand:
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})
        # view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
        if show_ligand_surface:
            view.addSurface(py3Dmol.VDW, {'opacity': 0.8}, {'model': -1})

    view.zoomTo()
    return view


def visualize_complex_with_frags(pdb_block, all_frags, show_protein_surface=True, show_ligand=True, show_ligand_surface=True):
    view = py3Dmol.view()

    # Add protein to the canvas
    view.addModel(pdb_block, 'pdb')
    if show_protein_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
    else:
        view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0}})

    # Add ligand to the canvas
    if show_ligand:
        for frag in all_frags:
            sdf_block = Chem.MolToMolBlock(frag)
            view.addModel(sdf_block, 'sdf')
            view.setStyle({'model': -1}, {'stick': {}})
            # view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
            if show_ligand_surface:
                view.addSurface(py3Dmol.VDW, {'opacity': 0.8}, {'model': -1})

    view.zoomTo()
    return view


def visualize_complex_highlight_pocket(pdb_block, sdf_block,
                                       pocket_atom_idx, pocket_res_idx=None, pocket_chain=None,
                                       show_ligand=True, show_ligand_surface=True):
    view = py3Dmol.view()

    # Add protein to the canvas
    view.addModel(pdb_block, 'pdb')
    view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
    if pocket_atom_idx:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'red'}, {'model': -1, 'serial': pocket_atom_idx})
    if pocket_res_idx:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'red'},
                        {'model': -1, 'chain': pocket_chain, 'resi': list(set(pocket_res_idx))})
    # color_map = ['red', 'yellow', 'blue', 'green']
    # for idx, pocket_atom_idx in enumerate(all_pocket_atom_idx):
    #     print(pocket_atom_idx)
    #     view.addSurface(py3Dmol.VDW, {'opacity':0.7, 'color':color_map[idx]}, {'model': -1, 'serial': pocket_atom_idx})
    # view.addSurface(py3Dmol.VDW, {'opacity':0.7,'color':'red'}, {'model': -1, 'resi': list(set(pocket_residue))})

    # view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0.}})
    # view.setStyle({'model': -1, 'serial': atom_idx},  {'cartoon': {'color': 'red'}})
    # view.setStyle({'model': -1, 'resi': [482, 484]},  {'cartoon': {'color': 'green'}})

    # Add ligand to the canvas
    if show_ligand:
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})
        # view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
        if show_ligand_surface:
            view.addSurface(py3Dmol.VDW, {'opacity': 0.8}, {'model': -1})

    view.zoomTo()
    return view


def visualize_mol_highlight_fragments(mol, match_list):
    all_target_atm = []
    for match in match_list:
        target_atm = []
        for atom in mol.GetAtoms():
            if atom.GetIdx() in match:
                target_atm.append(atom.GetIdx())
        all_target_atm.append(target_atm)

    return Draw.MolsToGridImage([mol for _ in range(len(match_list))], highlightAtomLists=all_target_atm,
                                subImgSize=(400, 400), molsPerRow=4)


# doesn't work for now
def visualize_xyz_animation(atom_pos_traj, atom_type_traj, protein_path):
    ptable = Chem.GetPeriodicTable()
    with open(protein_path, 'r') as f:
        pdb_block = f.read()

    view = py3Dmol.view()
    models = ""
    for idx, (atom_pos, atom_type) in enumerate(zip(atom_pos_traj, atom_type_traj)):
        models += "MODEL " + str(idx) + "\n"
        num_atoms = len(atom_pos)
        xyz = "%d\n\n" % (num_atoms,)
        for i in range(num_atoms):
            symb = ptable.GetElementSymbol(atom_type[i])
            x, y, z = atom_pos[i]
            xyz += "%s %.8f %.8f %.8f\n" % (symb, x, y, z)
        models += xyz
        models += "ENDMDL\n"

    # Generated molecule
    view.addModelsAsFrames(models, 'xyz')
    view.setStyle({'model': list(range(len(atom_pos_traj)))}, {'sphere': {'radius': 0.3}, 'stick': {}})

    # Pocket
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})

    # Focus on the generated
    view.zoomTo()
    view.animate({'loop': "forward", 'reps': 2})
    return view


def visualize_generated_xyz_v2(atom_pos, atom_type, protein_path, ligand_path=None,
                               pocket_atom_idx_list=None, pocket_centers=None,
                               show_ligand=False, show_protein_surface=True, center_opacity=1.0):
    ptable = Chem.GetPeriodicTable()

    num_atoms = len(atom_pos)
    xyz = "%d\n\n" % (num_atoms,)
    for i in range(num_atoms):
        symb = ptable.GetElementSymbol(atom_type[i])
        x, y, z = atom_pos[i]
        xyz += "%s %.8f %.8f %.8f\n" % (symb, x, y, z)

    # print(xyz)

    with open(protein_path, 'r') as f:
        pdb_block = f.read()

    view = py3Dmol.view()
    # Generated molecule
    view.addModel(xyz, 'xyz')
    view.setStyle({'model': -1}, {'sphere': {'radius': 0.3}, 'stick': {}})

    # Pocket
    view.addModel(pdb_block, 'pdb')
    if show_protein_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
    else:
        view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0}})

    colors = ['red', 'blue', 'green', 'orange']
    if pocket_atom_idx_list:
        for i, pocket_atom_idx in enumerate(pocket_atom_idx_list):
            view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': colors[i % len(colors)]},
                            {'model': -1, 'serial': pocket_atom_idx})
    if pocket_centers is not None:
        for i, center in enumerate(pocket_centers):
            view.addSphere({'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
                            'color': colors[i % len(colors)], 'radius': 1., 'opacity': center_opacity})

    # Focus on the generated
    view.zoomTo()

    # Ligand
    if show_ligand:
        with open(ligand_path, 'r') as f:
            sdf_block = f.read()
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})

    return view


def visualize_generated_xyz(data, root, show_ligand=False):
    ptable = Chem.GetPeriodicTable()

    num_atoms = data.ligand_context_element.size(0)
    xyz = "%d\n\n" % (num_atoms,)
    for i in range(num_atoms):
        symb = ptable.GetElementSymbol(data.ligand_context_element[i].item())
        x, y, z = data.ligand_context_pos[i].clone().cpu().tolist()
        xyz += "%s %.8f %.8f %.8f\n" % (symb, x, y, z)

    # print(xyz)

    protein_path = os.path.join(root, data.protein_filename)
    ligand_path = os.path.join(root, data.ligand_filename)

    with open(protein_path, 'r') as f:
        pdb_block = f.read()
    with open(ligand_path, 'r') as f:
        sdf_block = f.read()

    view = py3Dmol.view()
    # Generated molecule
    view.addModel(xyz, 'xyz')
    view.setStyle({'model': -1}, {'sphere': {'radius': 0.3}, 'stick': {}})
    # Focus on the generated
    view.zoomTo()

    # Pocket
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    # Ligand
    if show_ligand:
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})

    return view


def visualize_generated_sdf(data, protein_path, ligand_path, show_ligand=False, show_protein_surface=True):
    # protein_path = os.path.join(root, data.protein_filename)
    # ligand_path = os.path.join(root, data.ligand_filename)

    with open(protein_path, 'r') as f:
        pdb_block = f.read()

    view = py3Dmol.view()
    # Generated molecule
    mol_block = Chem.MolToMolBlock(data.rdmol)
    view.addModel(mol_block, 'sdf')
    view.setStyle({'model': -1}, {'sphere': {'radius': 0.3}, 'stick': {}})
    # Focus on the generated
    # view.zoomTo()

    # Pocket
    view.addModel(pdb_block, 'pdb')
    if show_protein_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
        view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0}})
    else:
        view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    # Ligand
    if show_ligand:
        with open(ligand_path, 'r') as f:
            sdf_block = f.read()
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})
    view.zoomTo()
    return view


def visualize_generated_arms(data_list, protein_path, ligand_path, show_ligand=False, show_protein_surface=True):
    # protein_path = os.path.join(root, data.protein_filename)
    # ligand_path = os.path.join(root, data.ligand_filename)

    with open(protein_path, 'r') as f:
        pdb_block = f.read()

    view = py3Dmol.view()
    # Generated molecule
    for data in data_list:
        mol_block = Chem.MolToMolBlock(data.rdmol)
        view.addModel(mol_block, 'sdf')
        view.setStyle({'model': -1}, {'sphere': {'radius': 0.3}, 'stick': {}})
    # Focus on the generated
    # view.zoomTo()

    # Pocket
    view.addModel(pdb_block, 'pdb')
    if show_protein_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'model': -1})
        view.setStyle({'model': -1}, {"cartoon": {"style": "edged", 'opacity': 0}})
    else:
        view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}, 'line': {}})
    # Ligand
    if show_ligand:
        with open(ligand_path, 'r') as f:
            sdf_block = f.read()
        view.addModel(sdf_block, 'sdf')
        view.setStyle({'model': -1}, {'stick': {}})
    view.zoomTo()
    return view


def visualize_ligand(mol, size=(300, 300), style="stick", surface=False, opacity=0.5):
    """Draw molecule in 3D

    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'carton')

    viewer = py3Dmol.view(width=size[0], height=size[1])
    if isinstance(mol, list):
        for i, m in enumerate(mol):
            mblock = Chem.MolToMolBlock(m)
            viewer.addModel(mblock, 'mol' + str(i))
    elif len(mol.GetConformers()) > 1:
        for i in range(len(mol.GetConformers())):
            mblock = Chem.MolToMolBlock(mol, confId=i)
            viewer.addModel(mblock, 'mol' + str(i))
    else:
        mblock = Chem.MolToMolBlock(mol)
        viewer.addModel(mblock, 'mol')
    viewer.setStyle({style: {}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer


def mol_with_atom_index(mol):
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def vis_decomp_from_data(data):
    with open(data.meta_file, 'rb') as f:
        r = pickle.load(f)
    vis_sub_mols = []
    for submol in r['data']['all_submols']:
        if submol is not None:
            vis_submol = copy.deepcopy(submol)
            vis_submol.RemoveAllConformers()
            vis_sub_mols.append(vis_submol)
    img = MolsToGridImage([submol for submol in vis_sub_mols], subImgSize=(400, 400), molsPerRow=4)
    return img


def vis_complex_from_data(data, **kwargs):
    with open(data.protein_file, 'r') as f:
        pdb_block = f.read()
    with open(data.ligand_file, 'r') as f:
        sdf_block = f.read()
    view = visualize_complex(pdb_block, sdf_block, **kwargs)
    return view


def vis_complex_with_decomp_centers(data, arm_centers, scaffold_center):
    with open(data.protein_file, 'r') as f:
        pdb_block = f.read()
    sdf_block = Chem.MolToMolBlock(data.ligand_rdmol)
    viewer = visualize_complex(pdb_block, sdf_block, show_ligand_surface=False)

    color_map = ['red', 'green', 'blue']
    for idx, c in enumerate(arm_centers):
        if torch.is_tensor(c):
            c = c.numpy().tolist()
        else:
            c = c.tolist()
        viewer.addSphere({'center': {'x': c[0], 'y': c[1], 'z': c[2]}, 'radius': 1.0, 'color': color_map[idx % 3]})
    if len(scaffold_center) > 0:
        s = scaffold_center[0]
        if torch.is_tensor(s):
            s = s.numpy().tolist()
        else:
            s = s.tolist()
        viewer.addSphere({'center': {'x': s[0], 'y': s[1], 'z': s[2]}, 'radius': 1.0, 'color': 'yellow'})
    return viewer


def vis_gen_complex_with_decomp_centers(mol, protein_file, arm_centers, scaffold_center):
    with open(protein_file, 'r') as f:
        pdb_block = f.read()
    sdf_block = Chem.MolToMolBlock(mol)
    viewer = visualize_complex(pdb_block, sdf_block, show_ligand_surface=False)

    color_map = ['red', 'green', 'blue']
    for idx, c in enumerate(arm_centers):
        if torch.is_tensor(c):
            c = c.numpy().tolist()
        else:
            c = c.tolist()
        viewer.addSphere({'center': {'x': c[0], 'y': c[1], 'z': c[2]}, 'radius': 1.0, 'color': color_map[idx % 3]})
    s = scaffold_center[0]
    if torch.is_tensor(s):
        s = s.numpy().tolist()
    else:
        s = s.tolist()
    viewer.addSphere({'center': {'x': s[0], 'y': s[1], 'z': s[2]}, 'radius': 1.0, 'color': 'yellow'})
    return viewer