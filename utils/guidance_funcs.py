# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


from collections import defaultdict

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_scatter import scatter_min

import utils.reconstruct as recon
import utils.transforms as trans
from utils.chem import ff_optimize, get_ring_systems


# Ligand residue locations: a_i in R^3. Receptor: b_j in R^3
# Ligand: G_l(x) = -sigma * ln( \sum_i  exp(- ||x - a_i||^2 / sigma)  ), same for G_r(x)
# Ligand surface: x such that G_l(x) = surface_ct
# Other properties: G_l(a_i) < 0, G_l(x) = infinity if x is far from all a_i
# Intersection of ligand and receptor: points x such that G_l(x) < surface_ct && G_r(x) < surface_ct
# Intersection loss: IL = \avg_i max(0, surface_ct - G_r(a_i)) + \avg_j max(0, surface_ct - G_l(b_j))
def G_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) , x: (m,3), output: (m,)
    e = torch.exp(-torch.sum((protein_coords.view(1, -1, 3) - x.view(-1, 1, 3)) ** 2, dim=2) / float(sigma))  # (m, n)
    return -sigma * torch.log(1e-3 + e.sum(dim=1))


def compute_body_intersection_loss(protein_coords, ligand_coords, sigma, surface_ct):
    loss = torch.mean(torch.clamp(surface_ct - G_fn(protein_coords, ligand_coords, sigma), min=0))
    return loss


def compute_batch_clash_loss(protein_pos, pred_ligand_pos, batch_protein, batch_ligand, sigma=25, surface_ct=10):
    loss_clash = torch.tensor(0., device=protein_pos.device)
    num_graphs = batch_ligand.max().item() + 1
    for i in range(num_graphs):
        p_pos = protein_pos[batch_protein == i]
        l_pos = pred_ligand_pos[batch_ligand == i]
        loss_clash += compute_body_intersection_loss(p_pos, l_pos, sigma=sigma, surface_ct=surface_ct)
    return loss_clash


def compute_center_prox_loss(pred_ligand_pos, noise_centers):
    loss = torch.norm(pred_ligand_pos - noise_centers, p=2, dim=-1)
    return loss


def compute_armsca_prox_loss(arm_pos, sca_pos, arm_index, min_d=1.2, max_d=1.9):
    pairwise_dist = torch.norm(arm_pos.unsqueeze(1) - sca_pos.unsqueeze(0), p=2, dim=-1)
    min_dist_all, _ = scatter_min(pairwise_dist, arm_index, dim=0)
    min_dist, min_dist_sca_idx = min_dist_all.min(-1)
    # loss_armsca_prox += torch.mean(pairwise_dist)
    # 1.2 < min dist < 1.9
    loss = torch.mean(torch.clamp(min_d - min_dist, min=0) + torch.clamp(min_dist - max_d, min=0))
    return loss


def compute_batch_armsca_prox_loss(pred_ligand_pos, batch_ligand, ligand_decomp_index, min_d=1.2, max_d=1.9):
    batch_losses = torch.tensor(0., device=pred_ligand_pos.device)
    num_graphs = batch_ligand.max().item() + 1
    n_valid = 0
    # ligand_pos.requires_grad = True
    for i in range(num_graphs):
        pos = pred_ligand_pos[batch_ligand == i]
        mask = ligand_decomp_index[batch_ligand == i]  # -1: scaffold, 0-N: arms
        arm_mask = (mask != -1)
        arm_pos = pos[arm_mask]
        sca_pos = pos[~arm_mask]
        if len(arm_pos) > 0 and len(sca_pos) > 0:
            loss = compute_armsca_prox_loss(arm_pos, sca_pos, mask[arm_mask], min_d=min_d, max_d=max_d)
            batch_losses += loss
            n_valid += 1
            # energy_grad = torch.autograd.grad(energy, ligand_pos)[0]
            # batch_grads += energy_grad
    # pos_model_mean -= energy_grad
    return batch_losses / num_graphs, n_valid


def compute_arms_repul_loss(arm1_pos, arm2_pos, max_d=1.9, mode='min'):
    pairwise_dist = torch.norm(arm1_pos.unsqueeze(1) - arm2_pos.unsqueeze(0), p=2, dim=-1)
    if mode == 'min':
        min_dist = pairwise_dist.min()
        loss = torch.mean(torch.clamp(max_d - min_dist, min=0))
    elif mode == 'all':
        # min dist > 1.9
        loss = torch.mean(torch.clamp(max_d - pairwise_dist, min=0))
    else:
        raise ValueError
    return loss


def compute_batch_arms_repul_loss(pred_ligand_pos, batch_ligand, ligand_decomp_index, max_d=1.9, mode='min'):
    batch_losses = torch.tensor(0., device=pred_ligand_pos.device)
    num_graphs = batch_ligand.max().item() + 1
    n_valid = 0
    # ligand_pos.requires_grad = True
    for i in range(num_graphs):
        pos = pred_ligand_pos[batch_ligand == i]
        mask = ligand_decomp_index[batch_ligand == i]  # -1: scaffold, 0-N: arms
        num_arms = mask.max().item() + 1
        for a1 in range(num_arms):
            for a2 in range(a1, num_arms):

                arm1_mask = (mask == a1)
                arm2_mask = (mask == a2)
                arm1_pos = pos[arm1_mask]
                arm2_pos = pos[arm2_mask]

                if len(arm1_pos) > 0 and len(arm2_pos) > 0:
                    loss = compute_arms_repul_loss(arm1_pos, arm2_pos, max_d=max_d, mode=mode)
                    batch_losses += loss
                    n_valid += 1
                    # energy_grad = torch.autograd.grad(energy, ligand_pos)[0]
                    # batch_grads += energy_grad
    # pos_model_mean -= energy_grad
    return batch_losses / num_graphs, n_valid


def compute_conf_drift(pred_ligand_pos, pred_ligand_v, batch_ligand, atom_enc_mode='add_aromatic', verbose=False):
    if not verbose:
        RDLogger.DisableLog('rdApp.*')

    num_graphs = batch_ligand.max().item() + 1
    batch_pred_pos = pred_ligand_pos.cpu().numpy().astype(np.float64)
    batch_pred_v = pred_ligand_v.cpu().numpy()
    batch_ligand = batch_ligand.cpu().numpy()
    pos_grad = []
    for i in range(num_graphs):
        pred_pos = batch_pred_pos[batch_ligand == i]
        pred_v = batch_pred_v[batch_ligand == i]
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)
        # first try to reconstruct mol, would be better if we also predict bond
        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode=atom_enc_mode)
            mol = recon.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
        except recon.MolReconsError:
            # print(f'Reconstruct failed at {i}')
            pos_grad.append(torch.zeros(pred_pos.shape))
            continue

        smiles = Chem.MolToSmiles(mol)
        if '.' in smiles:
            pos_grad.append(torch.zeros(pred_pos.shape))
            continue

        r = ff_optimize(mol, addHs=True)
        if r[0]:
            ff_mol = r[-1]
            ff_pos = ff_mol.GetConformer().GetPositions()
            grad = pred_pos - ff_pos
            pos_grad.append(torch.from_numpy(grad.astype(np.float32)))
        else:
            pos_grad.append(torch.zeros(pred_pos.shape))
    pos_grad = torch.cat(pos_grad, dim=0).to(pred_ligand_pos)
    assert pos_grad.shape == pred_ligand_pos.shape
    return pos_grad


def compute_ring_repulsion_drift(pred_ligand_pos, batch_ligand, bond_d=1.9, max_allow_rings=2, verbose=False):
    if not verbose:
        RDLogger.DisableLog('rdApp.*')

    batch_losses = torch.tensor(0., device=pred_ligand_pos.device)
    num_graphs = batch_ligand.max().item() + 1
    n_valid = 0
    # batch_pred_pos = pred_ligand_pos
    # batch_pred_v = pred_ligand_v.cpu().numpy()
    # batch_ligand = batch_ligand.cpu().numpy()

    for i in range(num_graphs):
        pred_pos = pred_ligand_pos[batch_ligand == i]
        # pred_v = batch_pred_v[batch_ligand == i]
        # pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)
        dummy_pred_atom_type = [6] * len(pred_pos)

        all_pairwise_dist = torch.norm(pred_pos.unsqueeze(1) - pred_pos.unsqueeze(0), p=2, dim=-1)
        all_bond_mask = (all_pairwise_dist < bond_d)
        all_bond_mask = all_bond_mask & (~torch.eye(all_bond_mask.size(0)).to(all_bond_mask))
        # if we have bond diffusion, use generated bond type
        bond_index = all_bond_mask.nonzero().T
        bond_type = [1] * bond_index.size(1)
        mol = recon.reconstruct_from_generated_with_bond(
            pred_pos, dummy_pred_atom_type, bond_index, bond_type, check_validity=False)

        Chem.GetSymmSSSR(mol)
        ri = mol.GetRingInfo()
        num_atom_rings = []
        for idx in range(mol.GetNumAtoms()):
            num_atom_rings.append(ri.NumAtomRings(idx))
        num_atom_rings = torch.tensor(num_atom_rings, device=pred_ligand_pos.device)
        fused_rings = get_ring_systems(mol)

        ringsys_max_distance = defaultdict(list)
        ringsys_num = defaultdict(int)
        for ring in ri.AtomRings():
            # fr_idx = [idx for idx, fr in enumerate(fused_rings) if len(set(fr).intersection(set(ring))) == len(ring)]
            fr_idx = [idx for idx, fr in enumerate(fused_rings) if ring[0] in fr and ring[1] in fr]
            assert len(fr_idx) == 1
            fr_idx = fr_idx[0]

            ring = torch.tensor(ring)
            bond_mask = all_bond_mask[ring][:, ring]
            # only 2ring-1ring / 1ring-1ring bond are considered
            cand_mask = (num_atom_rings[ring][:, None] + num_atom_rings[ring][None, :]) < 4

            pairwise_dist = torch.norm(pred_pos[ring].unsqueeze(1) - pred_pos[ring].unsqueeze(0), p=2, dim=-1)
            # print(pairwise_dist.shape, pairwise_dist)
            # print(pairwise_dist[bond_mask & cand_mask])
            cand_dist = pairwise_dist[bond_mask & cand_mask]
            if len(cand_dist) > 0:
                max_dist, max_dist_idx = cand_dist.max(-1)
                ringsys_max_distance[fr_idx].append(max_dist)
            ringsys_num[fr_idx] += 1

        repulsion_distances = []
        for k, v in ringsys_max_distance.items():
            num_sys = ringsys_num[k]
            v = torch.stack(v)
            if num_sys > max_allow_rings:
                if len(v) > num_sys - max_allow_rings:
                    v, _ = v.topk(num_sys - max_allow_rings)
                repulsion_distances.append(v)
        if len(repulsion_distances) > 0:
            repulsion_distances = torch.cat(repulsion_distances)
            loss = torch.mean(torch.clamp(bond_d - repulsion_distances, min=0.))
            batch_losses += loss
            n_valid += 1
    return batch_losses / num_graphs, n_valid
