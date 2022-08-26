import collections
import json
import math
import os
import os.path as osp
import sys
from multiprocessing import Process

import networkx as nx
import numpy as np
import pickle5 as pickle
import scipy as sp
import torch
import torch.nn as nn
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.data import Data

from common_models.models import models


def from_networkx(G, group_node_attrs=None, group_edge_attrs=None):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.
    """

    G = G.to_directed() if not nx.is_directed(G) else G

    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        edges = list(G.edges(keys=False))
    else:
        edges = list(G.edges)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = collections.defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except ValueError:
                pass

    data['edge_index'] = pyg.utils.sort_edge_index(edge_index.view(2, -1))
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data


def mlp_to_network(neuron_network, draw=False):
    """mlp_to_network

    Args:
        neuron_network (dict): key is layer name, and value is a 2D
            matrices
        draw (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    # nodes_num = 0
    graphs = collections.defaultdict(dict)
    # nodes_num=nodes_num+(neuron_network.conv.in_channels)*(neuron_network.conv.out_channels)
    # node_list.append(len(node_list)+np.arange(neuron_network.conv.in_channels + neuron_network.conv.out_channels))

    for l, (n, w) in enumerate(neuron_network.items()):
        nx_new = nx.Graph()
        if len(w.shape) == 4:  # c_out x c_in x kH x kw
            c_out, c_in, kH, kW = w.shape
            w = w.reshape(c_out, -1)
            w = np.transpose(w)
            assert w.shape[1] == c_out
        dim_in, dim_out = w.shape
        if l == 0:
            idx_in_start = 0
            idx_out_start = dim_in
        else:
            idx_in_start = idx_out_start
            idx_out_start = idx_out_start + dim_in
        for i in range(dim_in):
            nx_new.add_node(i)
            for j in range(dim_out):
                idx_in = i
                idx_out = j + dim_in
                nx_new.add_node(idx_out)
                edge_w = np.abs(w[i, j])
                if edge_w > 0:
                    nx_new.add_weighted_edges_from([(idx_in, idx_out, edge_w)])

        graphs[n]['idx_in_start'] = idx_in_start
        graphs[n]['idx_out_start'] = idx_out_start
        graphs[n]['dim_in'] = dim_in
        graphs[n]['dim_out'] = dim_out
        graphs[n]['sparsity'] = nx_new.number_of_edges() / (dim_in * dim_out)
        group_edge_attrs = None
        if len(nx_new.edges()) > 0:
            group_edge_attrs = ['weight']
        graphs[n]['graph'] = from_networkx(nx_new,
                                           group_edge_attrs=group_edge_attrs)
        graphs[n]['graph'].num_nodes = dim_in + dim_out

    return graphs


def generate_bipartie_graphs(m: nn.Module) -> dict:
    neuron_network = {}
    for name, module in m.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            neuron_network[name] = module.weight.detach().numpy()
    return mlp_to_network(neuron_network)


def process(path, model, num_classes, dst):
    print(f'working on {path}')
    m = models[model](num_classes)
    m.load_state_dict(torch.load(path, map_location='cpu'), strict=False)

    graphs = generate_bipartie_graphs(m)
    torch.save(graphs, osp.join(dst, osp.basename(path)))


QUEUE = 10
if __name__ == "__main__":
    _, path, dst = sys.argv
    os.makedirs(dst, exist_ok=True)
    files = os.listdir(path)
    files = [
        osp.join(path, f)
        for f in list(filter(lambda x: "mask.pth" in x, files))
    ]

    with open(osp.join(path, 'config.txt'), 'r') as file:
        config = json.load(file)
    model = config.get('model', config['arch'])
    dataset = config.get('dataset', config['data'])
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100

    for i in range(0, len(files), QUEUE):
        jobs = []
        for j in range(min(QUEUE, len(files) - i)):
            # process(files[i + j], model, num_classes, dst)
            jobs.append(
                Process(target=process,
                        args=(
                            files[i + j],
                            model,
                            num_classes,
                            dst,
                        )))
            jobs[-1].start()
        for job in jobs:
            job.join()
