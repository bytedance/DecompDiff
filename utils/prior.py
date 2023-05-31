# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import torch
import pickle
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import torch.nn.functional as F


def calculate_isotropic_covariance(pos):
    assert len(pos.shape) == 2
    assert pos.shape[1] == 3
    mu = pos.mean(axis=0, keepdims=True)
    pos = pos - mu  # pos [N,3]
    pos = pos.reshape(-1, 1)
    M = pos.shape[0]
    covariance = np.matmul(pos.T, pos)
    covariance = covariance / M * np.eye(3)
    return covariance


def calcula_anisotropic_covariance(pos):
    assert len(pos.shape) == 2
    assert pos.shape[1] == 3
    mu = pos.mean(axis=0, keepdims=True)
    pos = pos - mu  # pos [N,3]
    N = pos.shape[0]
    covariance = np.matmul(pos.T, pos)
    covariance = covariance / N
    return covariance


def get_iso_aniso_mu_cov(pos):
    if pos.shape[0] == 0:
        iso_mu = np.zeros_like(pos)
        iso_cov = np.eye(0)
        aniso_mu = np.zeros_like(pos)
        aniso_cov = np.eye(0)
    else:
        iso_mu = aniso_mu = pos.mean(axis=0)
        iso_cov = calculate_isotropic_covariance(pos)
        aniso_cov = calcula_anisotropic_covariance(pos)
    return iso_mu, iso_cov, aniso_mu, aniso_cov


def substitute_golden_prior_with_beta_prior(data, beta_prior_path, protein_ligand_dist_th=10.0):
    beta_prior = pickle.load(open(beta_prior_path, 'rb'))
    data.num_arms = len(beta_prior['arms_prior'])
    data.num_scaffold = len(beta_prior['scaffold_prior'])
    assert data.num_arms == beta_prior['num_arms']
    assert data.num_scaffold == beta_prior['num_scaffold']

    data.arms_prior = []
    data.scaffold_prior = []
    data.pocket_atom_masks = []
    for arms_prior in beta_prior['arms_prior']:
        num, mu_i, cov_i, mu_a, cov_a = arms_prior
        data.arms_prior.append((num, torch.tensor(mu_i).float(), torch.tensor(cov_i).float(), None, None))
        mu_i = torch.tensor(mu_i).float().reshape(1, 3)
        dist = pairwise_distances(mu_i, data.protein_pos).reshape(-1)
        mask = dist < protein_ligand_dist_th
        data.pocket_atom_masks.append(mask)
    if len(beta_prior['scaffold_prior']) == 1:
        num, mu_i, cov_i, mu_a, cov_a = beta_prior['scaffold_prior'][0]
        data.scaffold_prior.append((num, torch.tensor(mu_i).float(), torch.tensor(cov_i).float(), None, None))
    data.pocket_atom_masks = torch.tensor(data.pocket_atom_masks)


def substitute_golden_prior_with_given_prior(data, prior_dict, protein_ligand_dist_th=10.0):
    data.num_arms = len(prior_dict['arms_prior'])
    data.num_scaffold = len(prior_dict['scaffold_prior'])
    assert len(prior_dict['scaffold_prior']) <= 1

    data.arms_prior = []
    data.scaffold_prior = []
    data.pocket_atom_masks = []
    for arms_prior in prior_dict['arms_prior']:
        num, mu_i, cov_i, mu_a, cov_a = arms_prior
        data.arms_prior.append((num, torch.tensor(mu_i).float(), torch.tensor(cov_i).float(), None, None))
        mu_i = torch.tensor(mu_i).float().reshape(1, 3)
        dist = pairwise_distances(mu_i, data.protein_pos).reshape(-1)
        mask = dist < protein_ligand_dist_th
        data.pocket_atom_masks.append(mask)
    if len(prior_dict['scaffold_prior']) == 1:
        num, mu_i, cov_i, mu_a, cov_a = prior_dict['scaffold_prior'][0]
        data.scaffold_prior.append((num, torch.tensor(mu_i).float(), torch.tensor(cov_i).float(), None, None))
    data.pocket_atom_masks = torch.tensor(data.pocket_atom_masks)


def apply_std_coef(data, std_coef):
    new_arms_prior = []
    for arm_prior in data.arms_prior:
        num, mu_i, cov_i, _, _ = arm_prior
        cov_i *= std_coef ** 2
        new_arms_prior.append((num, mu_i, cov_i, None, None))
    data.arms_prior = new_arms_prior
    new_scaffold_prior = []
    if len(data.scaffold_prior) > 0:
        assert len(data.scaffold_prior) == 1
        num, mu_i, cov_i, _, _ = data.scaffold_prior[0]
        cov_i *= std_coef ** 2
        new_scaffold_prior.append((num, mu_i, cov_i, None, None))
    data.scaffold_prior = new_scaffold_prior


def apply_num_atoms_change(data, num_atoms_change):
    new_arms_prior = []
    for arm_prior in data.arms_prior:
        num, mu_i, cov_i, _, _ = arm_prior
        num += num_atoms_change
        num = max(num, 1)
        new_arms_prior.append((num, mu_i, cov_i, None, None))
    data.arms_prior = new_arms_prior
    new_scaffold_prior = []
    if len(data.scaffold_prior) > 0:
        assert len(data.scaffold_prior) == 1
        num, mu_i, cov_i, _, _ = data.scaffold_prior[0]
        num += num_atoms_change
        num = max(num, 1)
        new_scaffold_prior.append((num, mu_i, cov_i, None, None))
    data.scaffold_prior = new_scaffold_prior


def compute_golden_prior_from_data(data):
    # add prior
    pocket_prior_masks = []
    pocket_prior_contact_threshold = 6.0
    # >>> arms
    arms_prior = []
    for arm_id in range(data.num_arms):
        arm_atom_pos = data.ligand_pos[data.ligand_atom_mask == arm_id, :]
        (arm_iso_mu, arm_iso_cov, arm_aniso_mu, arm_aniso_cov) = get_iso_aniso_mu_cov(arm_atom_pos)
        arm_atom_num = arm_atom_pos.shape[0]
        arms_prior.append((arm_atom_num, arm_iso_mu, arm_iso_cov, arm_aniso_mu, arm_aniso_cov))
        cdist = F.pairwise_distance(arm_iso_mu.unsqueeze(0), data.protein_pos)
        cmask = cdist < pocket_prior_contact_threshold
        pocket_prior_masks.append(cmask)
    # >>> scaffold
    scaffold_prior = []
    scaffold_atom_pos = data.ligand_pos[data.ligand_atom_mask == -1, :]
    (scaffold_iso_mu, scaffold_iso_cov, scaffold_aniso_mu,
     scaffold_aniso_cov) = get_iso_aniso_mu_cov(scaffold_atom_pos)
    scaffold_atom_num = scaffold_atom_pos.shape[0]
    if scaffold_atom_num > 0:
        scaffold_prior.append((scaffold_atom_num, scaffold_iso_mu, scaffold_iso_cov,
                               scaffold_aniso_mu, scaffold_aniso_cov))
        cdist = F.pairwise_distance(scaffold_iso_mu.unsqueeze(0), data.protein_pos)
        cmask = (cdist < pocket_prior_contact_threshold).bool()
        pocket_prior_masks.append(cmask)

    data.scaffold_prior = scaffold_prior
    data.arms_prior = arms_prior
    assert len(data.arms_prior) == data.num_arms
    assert len(data.scaffold_prior) == data.num_scaffold
    data.pocket_prior_masks = torch.stack(pocket_prior_masks)
    assert len(data.pocket_prior_masks) == data.num_arms + data.num_scaffold
    return data


class NumAtomsSampler():
    def __init__(self, pred_models_dict):
        super().__init__()
        self.arm_model = pred_models_dict['arm_model']
        self.armstd_model = pred_models_dict['armstd_model']
        self.sca_model = pred_models_dict['sca_model']
        self.scastd_model = pred_models_dict['scastd_model']

    def sample_arm_natoms(self, arm_centers, protein_pos):
        pair_distance = torch.norm(arm_centers.view(-1, 1, 3) - protein_pos.view(1, -1, 3), p=2, dim=-1)
        p_natoms = torch.stack([(pair_distance < r).sum(1) for r in np.linspace(1, 10, 50)], dim=1)
        x = p_natoms.numpy()
        y = self.arm_model.predict(x)
        # print('arm n atom prediction: ', y)
        arm_natoms = self.sample_natoms_from_prediction(y, std=0.2)
        arm_stds = self.armstd_model.predict(arm_natoms[:, None])
        arm_natoms = arm_natoms.tolist()
        arm_stds = torch.from_numpy(arm_stds.astype(np.float32)).reshape(-1, 1).expand(-1, 3)
        return arm_natoms, arm_stds

    def sample_sca_natoms(self, sca_center, arm_centers, arm_stds, protein_pos):
        pair_distance = torch.norm(sca_center.view(-1, 1, 3) - protein_pos.view(1, -1, 3), p=2, dim=-1)
        p_natoms = torch.stack([(pair_distance < r).sum(1) for r in np.linspace(1, 10, 50)], dim=1)

        armsca_distances = torch.norm(sca_center.view(-1, 1, 3) - arm_centers.view(1, -1, 3), p=2, dim=-1).numpy()
        # print('armsca_distances: ', armsca_distances)
        armsca_res = [d - r for d, r in zip(armsca_distances, arm_stds.numpy())]

        p_natoms_feat = p_natoms.numpy()
        # print(p_natoms_feat.shape)
        dist_feat = np.array([d.sum() for d in armsca_res])
        # print(dist_feat.shape)

        x = np.concatenate([p_natoms_feat, dist_feat[:, None]], axis=-1)
        y = self.sca_model.predict(x)
        # print('x shape: ', x.shape, y.shape)
        sca_natoms = self.sample_natoms_from_prediction(y, std=0.)
        sca_stds = self.scastd_model.predict(sca_natoms[:, None])
        assert len(sca_natoms) == len(sca_stds) == 1
        sca_natoms = sca_natoms.tolist()[0]
        sca_stds = torch.from_numpy(sca_stds.astype(np.float32)).expand(3)
        return sca_natoms, sca_stds

    def sample_natoms_from_prediction(self, n, std, min_natoms=2):
        natoms = np.ceil(n + std * n * np.random.randn(len(n))).astype(int)
        natoms = np.maximum(natoms, min_natoms)
        return natoms
