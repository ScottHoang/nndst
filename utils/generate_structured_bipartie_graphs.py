import collections
import json
import os
import os.path as osp
import sys
from multiprocessing import Process
from typing import Any
from typing import Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from generate_unstructured_bipartie_graphs import from_networkx
from generate_unstructured_bipartie_graphs import generate_nx_graphs
from generate_unstructured_bipartie_graphs import mlp_to_network as unstruct_mlp_to_network
from generate_unstructured_bipartie_graphs import update_collections
from set_eb_masks import pruning

from common_models.models import models


def mlp_to_network(neuron_network, draw=False):
    """mlp_to_network

    Args:
        neuron_network (dict): key is layer name, and value is a 2D
            matrices
        draw (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    graphs = collections.defaultdict(dict)

    idx_out_start = 0
    for l, (name, data) in enumerate(neuron_network.items()):
        weight, sub_mask = data
        nx_new, dim_in, dim_out, idx_in_start, idx_out_start = generate_nx_graphs(
            weight, l, idx_out_start, masks=sub_mask)
        graphs[name].update(
            update_collections(nx_new, idx_in_start, idx_out_start, dim_in,
                               dim_out))
    return graphs


def generate_bipartie_graphs(m: nn.Module,
                             masks: torch.Tensor) -> collections.defaultdict:
    neuron_network = {}
    c_idx = 0
    for name, module in m.named_modules():
        if isinstance(module, (nn.Conv2d)):
            weight = module.weight.detach()
            c_out, c_in, h, w = weight.shape
            sub_mask = masks[c_idx:c_idx + c_out]
            sub_mask = sub_mask.tile(c_in, 1).T.tile(h * w).reshape(
                c_out, c_in, h, w)
            neuron_network[name] = (weight * sub_mask).numpy()
            c_idx += c_out
    return unstruct_mlp_to_network(neuron_network)


# def generate_bipartie_graphs(m: nn.Module,
# masks: torch.Tensor) -> collections.defaultdict:
# neuron_network = {}
# c_idx = 0
# for name, module in m.named_modules():
# if isinstance(module, (nn.Conv2d)):
# weight = module.weight.detach().numpy()
# c_out = weight.shape[0]
# sub_mask = masks[c_idx:c_idx + c_out].tolist()
# neuron_network[name] = (weight, sub_mask)
# c_idx += c_out
# return mlp_to_network(neuron_network)


def process(path, model, num_classes, dst, ratio):
    print(f'working on {path}')
    m = models[model](num_classes)

    m.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    model_mask = pruning(m, ratio)
    print(f"model density: {model_mask.sum() / model_mask.size(0)}")
    graphs = generate_bipartie_graphs(m, model_mask)
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
    model = config['arch']
    dataset = config['dataset']
    ratio = 1 - config['sparsity']
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100

    for i in range(0, len(files), QUEUE):
        jobs = []
        for j in range(min(QUEUE, len(files) - i)):
            process(files[i + j], model, num_classes, dst, ratio)
            # jobs.append(
            # Process(target=process,
            # args=(files[i + j], model, num_classes, dst, ratio)))
            # jobs[-1].start()
        # for job in jobs:
        # job.join()
