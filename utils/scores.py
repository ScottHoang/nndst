"""
File: scores.py
Description: get various score criterias
"""
import math
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch_geometric as pyg


def get_eig_values(matrix: np.array) -> List[float]:
    """
    get the real eig of a square matrix
    """
    adj_eigh_val, _ = sp.linalg.eig(matrix)
    idx = adj_eigh_val.argsort()
    adj_eigh_val = [i.real for i in adj_eigh_val[idx]]
    return adj_eigh_val


def delta_s(eig_first: float, eig_second: float) -> float:
    """
    calc the change in eig based on https://proceedings.mlr.press/v162/pal22a.html
    """
    if eig_first < 1:
        return -1
    return (2 * math.sqrt(eig_first - 1) - eig_second) / eig_second


def delta_r(avg_deg_left: float, avg_deg_right: float,
            eig_second: float) -> float:
    """
    calc the change in degree based on https://proceedings.mlr.press/v162/pal22a.html
    """
    if avg_deg_left < 1 or avg_deg_right < 1:
        return -1
    return (math.sqrt(avg_deg_left - 1) + math.sqrt(avg_deg_right - 1) -
            eig_second) / eig_second


def ramanujan_score(layer: dict) -> Tuple[Tuple]:
    """
    folowing the methodology outline in https://proceedings.mlr.press/v162/pal22a.html
    computes the weighted and unweighted change in degree (del_r) and change in eigenvalues (del_s)
    """
    graph = layer['graph']
    degree = pyg.utils.degree(graph.edge_index[0], graph.num_nodes)
    d_avg_l = degree[0:layer['dim_in']].mean()
    d_avg_r = degree[layer['dim_in']::].mean()
    try:

        m_eig_vals = get_eig_values(
            pyg.utils.to_dense_adj(
                graph.edge_index,
                max_num_nodes=graph.num_nodes).squeeze().numpy())
        w_eig_vals = get_eig_values(
            pyg.utils.to_dense_adj(
                graph.edge_index,
                edge_attr=graph.edge_attr,
                max_num_nodes=graph.num_nodes).squeeze().numpy())

        t2_m, t1_m = m_eig_vals[-2], m_eig_vals[-1]
        t2_w, t1_w = w_eig_vals[-2], w_eig_vals[-1]

        del_s_m = delta_s(t1_m, t2_m)
        del_r_m = delta_r(d_avg_l, d_avg_r, t2_m)
        del_s_w = delta_s(t1_w, t2_w)
        del_r_w = delta_r(d_avg_l, d_avg_r, t2_w)

        return (del_s_m, del_r_m, t1_m), (del_s_w, del_r_w, t1_w)
    except:
        return (-1, -1, -1), (-1, -1, -1)


def pair_layers(layernames: List[str]) -> List[str]:
    """
    get sequential pairing layer for resnet type model
    params:
        layernames: list of names in already seq order in resnset.
    return pair of names
    """
    pairs = []
    for i in range(1, len(layernames)):
        cur = layernames[i]
        prev = layernames[i - 1]
        if cur == 'fc' or prev == 'fc':
            continue
        if 'downsample' not in cur:
            pairs.append([prev, cur])
            if 'downsample' in prev:
                pairs.append([pairs[-3][-1], cur])
        else:
            components = cur.split('.')
            sublayer = int(components[1])
            if sublayer == 0:
                for j in range(i - 1, -1, -1):
                    prev_layer_comp = layernames[j].split('.')
                    if prev_layer_comp[0] != components[0]:
                        pairs.append([layernames[j], cur])
                        break
            else:
                for j in range(i - 1, -1, -1):
                    prev_layer_comp = layernames[j].split('.')
                    if int(prev_layer_comp[1]) < sublayer:
                        pairs.append([layernames[j], cur])
                        break

    return pairs


def get_degrees(graph: pyg.data.Data) -> Union[torch.Tensor, None]:
    """get degrees of a graphs if there are edges in this graph

    :graphs: TODO
    :returns: TODO

    """
    edges = graph.edge_index
    if edges[0].size(0) > 0:
        return pyg.utils.degree(graph.edge_index[0])
    return None


def copeland_score(layer1: dict, layer2: dict) -> float:
    """
    get a copeland score. input and output node degree are normalized
    across input and output masks. this modified copeland scores in [0,1] and
    is the quotient between normalized in and out degree
    params:
        layer1: dict type produced by generate_bipartie_graph
        layer2: ~same type as layer1. And should be the sequential layer after layer1
    return:
        the modified copeland score
    """
    l1_out = layer1['dim_out']
    l2_in = layer2['dim_in']
    k_size = l2_in // l1_out
    l1deg = get_degrees(layer1['graph'])
    l2deg = get_degrees(layer2['graph'])

    if l1deg is None or l2deg is None:
        return 0.0
    in_deg = l1deg[-l1_out::]
    out_deg = l2deg[0:l2_in].reshape(l1_out, k_size)
    ###
    in_deg_norm = in_deg / (l1deg[0:layer1['dim_in']] != 0).sum()
    out_deg_norm = out_deg / (l2deg[layer2['dim_in']::] != 0).sum()
    ###
    mask = in_deg != 0
    ###
    out_deg_m = out_deg_norm[mask]
    in_deg_m = in_deg_norm[mask]
    throughput = out_deg_m / in_deg_m.tile(k_size).view(
        k_size, in_deg_m.size(0)).T
    return throughput.mean().item()


def channel_overlap_coefs(layer: dict, in_channels: int) -> float:
    """
    get the mean channel's kernel overlap coefs
    overlap coef is defined to be the ratio between intersection(A,B) / min(|A|, |B|)

    params:
        layer: the layer dictionary generated by generate_bipartie_graph.py
        in_channels: the input channels of l1's
    """
    k_size = layer['dim_in'] // in_channels
    channel_coefs = []
    for channel in range(in_channels):
        overlap_nodes = None
        min_graph = float("inf")
        for k in range(k_size):
            node = channel * k_size + k
            mask = layer['graph'].edge_index[0] == node
            tgt_nodes = set(layer['graph'].edge_index[1, mask].tolist())
            min_graph = min(min_graph, mask.float().sum().item())
            if overlap_nodes is None:
                overlap_nodes = tgt_nodes
            else:
                overlap_nodes = overlap_nodes.intersection(tgt_nodes)
        if min_graph == 0:
            continue
        coef = len(overlap_nodes) / min_graph
        channel_coefs.append(coef)
    if len(channel_coefs) > 0:
        return sum(channel_coefs) / len(channel_coefs)
    return 0


def compatibility_ratio(layer1: dict, layer2: dict) -> float:
    """Get iou of input / output degree between two layers.

    :layer1: TODO
    :layer2: TODO
    :returns: TODO

    """
    l1_out = layer1['dim_out']
    l2_in = layer2['dim_in']
    k_size = l2_in // l1_out
    l1deg = get_degrees(layer1['graph'])
    l2deg = get_degrees(layer2['graph'])

    if l1deg is None or l2deg is None:
        return 0
    ####
    in_deg = l1deg[-l1_out::]
    out_deg = l2deg[0:l2_in].reshape(l1_out, k_size).mean(dim=-1)
    ####
    in_mask = in_deg != 0.0
    out_mask = out_deg != 0.0
    compatibility = (in_mask
                     & out_mask).float().sum().item() / in_mask.sum().item()
    return compatibility


def ERK(layer1: dict, module: nn.Module) -> float:
    """TODO: Docstring for ERK.

    :layer1: TODO
    :module: TODO
    :returns: TODO

    """
    pass
