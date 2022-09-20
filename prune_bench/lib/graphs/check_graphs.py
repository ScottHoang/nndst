import json
import os
import os.path as osp
import sys

import torch
from generate_unstructured_bipartie_graphs import process as generate_unstructured_graphs
from get_otherscores import main as generate_scores
from get_ramanujanscores_per_timestep import process as generate_ram_score
from utils import link_latest


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


def check(file, dst, seed):

    tgt = file.replace("model.pth", "graph.pth")
    if not check_file(tgt):
        generate_unstructured_graphs(file, model, num_classes, tgt, seed)

    if not check_ram(tgt):
        print("generate ram score")
        generate_ram_score(tgt)

    if not check_others(tgt):
        print("generate other score")
        generate_scores(tgt)


if __name__ == "__main__":
    _, path = sys.argv
    ##
    prunes = "SNIP GraSP SynFlow ERK Rand iterSNIP" #PHEW"
    prunes = prunes.split(' ')
    # os.makedirs(dst, exist_ok=True)
    ##
    for p in prunes:
        if 'csv' in p:
            continue
        link_latest(osp.join(path, p))
        seeds = os.listdir(osp.join(path, p))
        for s in seeds:
            file = osp.join(path, p, str(s), 'latest', 'model.pth')

            with open(osp.join(path, p, str(s), 'latest', 'config.txt'),
                      'r') as handle:
                config = json.load(handle)

            dataset = config['data']
            model = config['model']
            sparsity = None

            dataset = config.get('dataset', config['data'])
            if dataset == 'cifar10':
                num_classes = 10
            elif dataset == 'cifar100':
                num_classes = 100

            check(file, osp.join(path, p, str(s), 'latest'), s)
