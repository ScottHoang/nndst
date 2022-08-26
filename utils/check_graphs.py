import json
import os
import os.path as osp
import sys

import torch
from generate_bipartie_graphs import process as generate_graphs
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

    for file in files:
        tgt = osp.join(dst, osp.basename(file))
        if not check_file(tgt):
            generate_graphs(file, model, num_classes, dst)

        if not check_ram(tgt):
            print("generate ram score")
            generate_ram_score(tgt)

        if not check_others(tgt):
            print("generate other score")
            generate_scores(tgt)
