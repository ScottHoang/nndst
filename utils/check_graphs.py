import json
import os
import os.path as osp
import sys

import torch
from generate_structured_bipartie_graphs import process as generate_structured_graphs
from generate_unstructured_bipartie_graphs import process as generate_unstructured_graphs
from get_otherscores import main as generate_scores
from get_ramanujanscores_per_timestep import process as generate_ram_score


def check_file(graph_path):
    """check whether the file exist and readable

    :tgt: TODO
    :returns: TODO

    """
    try:
        graph = torch.load(graph_path)
        print(f"{graph_path} is available")
        return True
    except:
        print(f"{graph_path} is not available")
        return False


def check_ram(graph_path):
    """check ram score

    :tgt: TODO
    :returns: TODO

    """
    graph = torch.load(graph_path)
    for name, info in graph.items():
        if 'ram_scores' not in info:
            return False
    return True


def check_others(graph_path):
    """check for other scores

    :graph_path: TODO
    :returns: TODO

    """
    graphs = torch.load(graph_path)
    for name, info in graphs.items():
        if 'fc' in name: continue
        if 'overlap_coefs' not in info:
            return False
    return True


if __name__ == "__main__":
    _, path, dst, is_eb = sys.argv
    is_eb = int(is_eb)
    os.makedirs(dst, exist_ok=True)
    files = os.listdir(path)
    files = [
        osp.join(path, f)
        for f in list(filter(lambda x: "mask.pth" in x, files))
    ]

    with open(osp.join(path, 'config.txt'), 'r') as file:
        config = json.load(file)

    if is_eb:
        dataset = config['dataset']
        model = config['arch']
        sparsity = 1 - config['sparsity']
    else:
        dataset = config['data']
        model = config['model']
        sparsity = None

    dataset = config.get('dataset', config['data'])
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100

    for file in files:
        tgt = osp.join(dst, osp.basename(file))
        if not check_file(tgt):
            if is_eb:
                generate_structured_graphs(file, model, num_classes, dst,
                                           sparsity)
            else:
                generate_unstructured_graphs(file, model, num_classes, dst)

        if not check_ram(tgt):
            print("generate ram score")
            generate_ram_score(tgt)

        if not check_others(tgt):
            print("generate other score")
            generate_scores(tgt)
