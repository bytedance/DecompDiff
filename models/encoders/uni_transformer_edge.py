# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_sum
from torch_geometric.nn import radius_graph, knn_graph, radius, knn
import time
from models.common import GaussianSmearing, MLP, get_h_dist, get_r_feat, batch_hybrid_edge_connection, outer_product, AngularEncoding
from torch_sparse import SparseTensor


class NodeUpdateLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim,
                 act_fn='relu', norm=True, out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        # self.r_feat_dim = r_feat_dim
        self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim * 2 + edge_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        kv_input = torch.cat([edge_feat, hi, hj], -1)

        # compute k
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute v
        v = self.hv_func(kv_input)
        e_w = e_w.view(-1, 1) if e_w is not None else 1.
        v = v * e_w

        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        # output = output + h
        return output


class BondUpdateLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, norm=True, act_fn='relu', include_h_node=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads

        self.distance_expansion = GaussianSmearing()
        self.angle_expansion = AngularEncoding()
        # attention key func
        kv_input_dim = input_dim + 20 * 2 + self.angle_expansion.get_out_dim(1)
        q_input_dim = input_dim
        if include_h_node:
            kv_input_dim += input_dim * 2
            q_input_dim += input_dim

        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(q_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.include_h_node = include_h_node

    @staticmethod
    def triplets(edge_index, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k->j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]
        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, h, h_bond, pos, bond_index):
        N, E = h.size(0), h_bond.size(0)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(bond_index, num_nodes=N)

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()  # (E, )

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)  # (E3, )

        r_feat = self.distance_expansion(dist)  # (E, 20)
        a_feat = self.angle_expansion(angle)  # (E3, ?)

        # multi-head attention
        hi, hj, hk = h[idx_i], h[idx_j], h[idx_k]
        h_bond_kj, h_bond_ji = h_bond[idx_kj], h_bond[idx_ji]
        r_feat_kj, r_feat_ji = r_feat[idx_kj], r_feat[idx_ji]

        if self.include_h_node:
            kv_input = torch.cat([h_bond_kj, r_feat_kj, r_feat_ji, a_feat, hk, hj], -1)  # whether to include hk, hj
            q_input = torch.cat([h_bond_ji, hi], -1)  # whether to include hi
        else:
            kv_input = torch.cat([h_bond_kj, r_feat_kj, r_feat_ji, a_feat], -1)
            q_input = h_bond_ji

        # compute k, v, q
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.hv_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        q = self.hq_func(q_input).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        alpha = scatter_softmax((q * k / np.sqrt(k.shape[-1])).sum(-1), idx_ji, dim=0)  # [E3, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E3, heads, H_per_head)
        output = scatter_sum(m, idx_ji, dim=0, dim_size=E)  # (E, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        # output = output + h_bond
        return output


class PosUpdateLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim,
                 act_fn='relu', norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        # self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn

        kv_input_dim = input_dim * 2 + edge_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, rel_x, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        kv_input = torch.cat([edge_feat, hi, hj], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        e_w = e_w.view(-1, 1) if e_w is not None else 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)   # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 r_min=0., r_max=10., include_h_node=False,
                 x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        # self.num_x2h = num_x2h
        # self.num_h2x = num_h2x
        # self.r2_min = r_min ** 2 if r_min >= 0 else -(r_min ** 2)
        # self.r2_max = r_max ** 2
        self.r_min, self.r_max = r_min, r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)
        self.lin_node = nn.Linear(hidden_dim, hidden_dim)
        self.node_layer_with_edge = NodeUpdateLayer(
            hidden_dim, hidden_dim, hidden_dim, n_heads,
            edge_feat_dim=num_r_gaussian * edge_feat_dim + edge_feat_dim,
            act_fn=act_fn, norm=norm, out_fc=self.x2h_out_fc
        )
        self.node_layer_with_bond = NodeUpdateLayer(
            hidden_dim, hidden_dim, hidden_dim, n_heads,
            edge_feat_dim=hidden_dim,
            act_fn=act_fn, norm=norm, out_fc=self.x2h_out_fc
        )
        self.bond_layer = BondUpdateLayer(
            hidden_dim, hidden_dim, hidden_dim, n_heads,
            act_fn=act_fn, norm=norm, include_h_node=include_h_node
        )
        self.pos_layer_with_edge = PosUpdateLayer(
            hidden_dim, hidden_dim, hidden_dim, n_heads,
            edge_feat_dim=num_r_gaussian * edge_feat_dim + edge_feat_dim,
            act_fn=act_fn, norm=norm,
        )
        self.pos_layer_with_bond = PosUpdateLayer(
            hidden_dim, hidden_dim, hidden_dim, n_heads,
            edge_feat_dim=hidden_dim,
            act_fn=act_fn, norm=norm,
        )

    def forward(self, h, x, edge_attr, edge_index, h_bond, bond_index, mask_ligand, e_w=None):

        # -> protein-ligand knn graph message passing
        src, dst = edge_index
        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        # --> 4 separate distance embedding for p-p, p-l, l-p, l-l
        dist_feat = self.distance_expansion(dist)
        dist_feat = outer_product(edge_attr, dist_feat)
        edge_feat = torch.cat([dist_feat, edge_attr], dim=-1)
        new_h_with_edge = self.node_layer_with_edge(h, edge_feat, edge_index, e_w=e_w)  # ht+1' = f(xt, ht)

        # -> ligand graph message passing
        new_h_with_bond = self.node_layer_with_bond(h, h_bond, bond_index)  # ht+1 '' = f(ht, bt, xt)
        new_h_bond = h_bond + self.bond_layer(h, h_bond, x, bond_index)  # bt+1 = h(ht, bt, xt)

        # --> update h node
        new_h = h + self.lin_node(new_h_with_edge + new_h_with_bond)

        # update x
        delta_x_with_edge = self.pos_layer_with_edge(new_h, rel_x, edge_feat, edge_index, e_w=e_w)   # g(xt, ht+1, ..)
        bond_src, bond_dst = bond_index
        rel_bond_x = x[bond_dst] - x[bond_src]
        delta_x_with_bond = self.pos_layer_with_bond(new_h, rel_bond_x, new_h_bond, bond_index)
        delta_x = delta_x_with_edge + delta_x_with_bond
        x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated

        return new_h, new_h_bond, x


class UniTransformerO2TwoUpdateGeneralBond(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32, num_bond_classes=1,
                 num_r_gaussian=50, edge_feat_dim=0, act_fn='relu', norm=True,
                 cutoff_mode='radius', use_global_ew=True,
                 r_max=10., x2h_out_fc=True, sync_twoup=False, h_node_in_bond_net=False):
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.k = k
        self.num_bond_classes = num_bond_classes

        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        self.use_global_ew = use_global_ew
        if self.use_global_ew:
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)
        self.h_node_in_bond_net = h_node_in_bond_net

        # self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, \n' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    # def _build_init_h_layer(self):
    #     layer = AttentionLayerO2TwoUpdateNodeGeneral(
    #         self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
    #         r_max=self.r_max,
    #         x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
    #     )
    #     return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
                r_max=self.r_max,
                x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup, include_h_node=self.h_node_in_bond_net
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    def _build_edge_type(self, edge_index, mask_ligand, decomp_group_idx):
        """
        Args:
            edge_index: (2, E)
            mask_ligand: (N, )
            decomp_group_idx: (N, )
        """
        # denote ll, lp, pl, pp edge type, prior dummy atom is considered as ligand atom
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        # edge_type[n_src & n_dst] = 0
        # edge_type[n_src & ~n_dst] = self.num_bond_classes
        # edge_type[~n_src & n_dst] = self.num_bond_classes + 1
        # edge_type[~n_src & ~n_dst] = self.num_bond_classes + 2
        # edge_type = F.one_hot(edge_type, num_classes=self.num_bond_classes + 3)

        if decomp_group_idx is not None:
            # denote whether protein / ligand / dummy are from the same prior
            decomp_type = torch.zeros(len(src)).to(edge_index)
            g_src = decomp_group_idx[src]
            g_dst = decomp_group_idx[dst]
            decomp_type[g_src == g_dst] = 1
            decomp_type = F.one_hot(decomp_type, num_classes=2)
            edge_type = torch.cat([edge_type, decomp_type], dim=-1)
        return edge_type

    def forward(self, h, x, group_idx, bond_index, h_bond, mask_ligand, mask_ligand_atom, batch, return_all=False):

        # full_src, full_dst = edge_index
        # h, _ = self.init_h_emb_layer(h, x)
        all_x = [x]
        all_h = [h]
        all_h_bond = [h_bond]

        for b_idx in range(self.num_blocks):
            # t1 = time.time()
            edge_index = self._connect_edge(x, mask_ligand, batch)
            # t2 = time.time()
            # print('edge index shape: ', edge_index.shape)
            # edge_length = torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=1)
            # edge_attr = self.distance_expansion(edge_length)

            # edge type (dim: 4)
            # print('edge index shape: ', edge_index.shape)
            edge_type = self._build_edge_type(edge_index, mask_ligand, group_idx)
            # if bond_index is not None:
            #     bond_edge_type = F.one_hot(bond_type, num_classes=self.num_bond_classes)
            #     pad_edge_type = torch.zeros([bond_type.size(0), edge_type.size(1) - self.num_bond_classes]).to(edge_type)
            #     bond_edge_type = torch.cat([bond_edge_type, pad_edge_type], dim=-1)
            #     edge_index = torch.cat([edge_index, bond_index], dim=1)
            #     edge_type = torch.cat([edge_type, bond_edge_type], dim=0)

            src, dst = edge_index
            # t3 = time.time()
            if self.use_global_ew:
                # dist = torch.sum((x[dst] - x[src]) ** 2, -1, keepdim=True)
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            for l_idx, layer in enumerate(self.base_block):
                h, h_bond, x = layer(h, x, edge_type, edge_index, h_bond, bond_index, mask_ligand_atom, e_w=e_w)
            # t4 = time.time()
            # print(f'connect edge time: {t2 - t1}, edge type compute time: {t3 - t2}, forward time: {t4 - t3}')
            all_x.append(x)
            all_h.append(h)
            all_h_bond.append(h_bond)

        # edge_index = self._connect_edge(x, mask_ligand, batch)
        outputs = {'x': x, 'h': h, 'h_bond': h_bond}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h, 'all_h_bond': all_h_bond})
        return outputs
