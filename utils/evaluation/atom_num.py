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


"""Utils for sampling size of a molecule of a given protein pocket."""

import numpy as np
from scipy import spatial as sc_spatial

from utils.evaluation.atom_num_config import CONFIG


def get_space_size(pocket_3d_pos):
    aa_dist = sc_spatial.distance.pdist(pocket_3d_pos, metric='euclidean')
    aa_dist = np.sort(aa_dist)[::-1]
    return np.median(aa_dist[:10])


def _get_bin_idx(space_size):
    bounds = CONFIG['bounds']
    for i in range(len(bounds)):
        if bounds[i] > space_size:
            return i
    return len(bounds)


def sample_atom_num(space_size, config_dict=None):
    bin_idx = _get_bin_idx(space_size)
    if config_dict is None:
        num_atom_list, prob_list = CONFIG['bins'][bin_idx]
    else:
        num_atom_list, prob_list = config_dict['bins'][bin_idx]
    return np.random.choice(num_atom_list, p=prob_list)
