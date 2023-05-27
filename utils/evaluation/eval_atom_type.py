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


from collections import Counter
from scipy import spatial as sci_spatial
import numpy as np

# ATOM_TYPE_DISTRIBUTION = {
#     6: 1585004,
#     7: 276248,
#     8: 400236,
#     9: 30871,
#     15: 26288,
#     16: 26529,
#     17: 15210,
# }

ATOM_TYPE_DISTRIBUTION = {
    6: 0.6715020339893559,
    7: 0.11703509510732567,
    8: 0.16956379168491933,
    9: 0.01307879304486639,
    15: 0.01113716146426898,
    16: 0.01123926340861198,
    17: 0.006443861300651673,
}


def eval_atom_type_distribution(pred_counter: Counter):
    total_num_atoms = sum(pred_counter.values())
    pred_atom_distribution = {}
    for k in ATOM_TYPE_DISTRIBUTION:
        pred_atom_distribution[k] = pred_counter[k] / total_num_atoms
    print('pred atom distribution: ', pred_atom_distribution)
    print('ref  atom distribution: ', ATOM_TYPE_DISTRIBUTION)
    js = sci_spatial.distance.jensenshannon(np.array(list(ATOM_TYPE_DISTRIBUTION.values())),
                                            np.array(list(pred_atom_distribution.values())))
    return js
