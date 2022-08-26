import os
import os.path as osp
import sys

import torch
from tqdm import tqdm

from scores import ramanujan_score as ram


def process(graph_path: str):
    """ for each graph get ram scores.
    """
    graphs = torch.load(graph_path)
    for (name, info) in tqdm(graphs.items(),
                             desc=osp.basename(graph_path),
                             total=len(graphs.keys())):
        if 'ram_scores' not in info:
            ram_scores = ram(info)
            info['ram_scores'] = ram_scores
    torch.save(graphs, graph_path)


if __name__ == "__main__":
    _, path = sys.argv
    files = os.listdir(path)
    files = [
        osp.join(path, f) for f in list(filter(lambda x: "mask" in x, files))
    ]

    for file in files:
        process(file)
