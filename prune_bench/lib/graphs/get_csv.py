import collections
import os
import os.path as osp
import sys
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
import torch
import torch_geometric as pyg
from scores import pair_layers
from scores import ramanujan_score
from tqdm import tqdm


def save(item: pd.DataFrame, dst: Optional[str], name: str):
    """
    save pandas df to csv file under name
    """
    if dst is not None:
        os.makedirs(dst, exist_ok=True)
        item.to_csv(osp.join(dst, name + '.csv'), index=False)


def get_summary_view(layers: pd.DataFrame) -> pd.DataFrame:
    """
    summarize the overall score of the entire masks by avg the values of each layer
    """
    summary = collections.defaultdict(list)
    for mask in layers.mask_no.unique():
        tab = layers.loc[layers.mask_no == mask]
        tab = tab.drop(columns=['layer'])
        summary['mask_no'].append(mask)
        for k in tab.keys():
            if k == 'mask_no':
                continue
            summary[k].append(tab[k].mean())

    return pd.DataFrame.from_dict(summary)


def find_related_pair(layer: str, pairs: List[Tuple]):
    """given name of a layer, finds its corresponding pairs

    :layer: TODO
    :pairs: TODO
    :returns: TODO

    """
    ret = []
    for p in pairs:
        if p[0][0] == layer and 'downsample' != p[-1][0]:
            return p
    return None


def generate_graph_csv(files: list, write=False) -> pd.DataFrame:
    """
    generate pandas Dataframe structure.
    params:
        TODO
    """
    layers_df = collections.defaultdict(list)
    for file in files:
        _, density, dataset, model, prune_type, seed, *_ = file.split('/')
        graphs = torch.load(file)
        pairs = pair_layers(list(graphs.items()))
        for i, (layer, info) in enumerate(graphs.items()):
            if 'ram_scores' in info:
                (s_m, r_m, t1m), (s_w, r_w, t1w) = info['ram_scores']
            else:
                (s_m, r_m, t1m), (s_w, r_w, t1w) = ramanujan_score(info)
            layers_df['prune_type'].append(prune_type)
            layers_df['layer'].append(layer)
            layers_df['sparsity'].append(info['sparsity'])
            layers_df['sm'].append(s_m)
            layers_df['rm'].append(r_m)
            layers_df['sw'].append(s_w)
            layers_df['rw'].append(r_w)
            layers_df['t1m'].append(t1m)
            layers_df['t1w'].append(t1w)

            related_pairs = find_related_pair(layer, pairs)
            if related_pairs:
                nxt = related_pairs[1][0]
                layers_df['copeland_score'].append(
                    info.get(f"{nxt}_copeland_score", 0))
                layers_df['compatibility'].append(
                    info.get(f"{nxt}_comatibility", 0))
                layers_df['overlap_coefs'].append(info.get('overlap_coefs', 0))
            else:
                layers_df['copeland_score'].append(0)
                layers_df['compatibility'].append(0)
                layers_df['overlap_coefs'].append(0)
    folder, density, dataset, model, prune_type, seed, *_ = files[0].split('/')
    name = f"graph_seed-{seed}"
    layers_df = pd.DataFrame.from_dict(layers_df)
    if write:
        save(layers_df, osp.join(folder, density, dataset, model, 'csv'), name)
    return layers_df


def generate_perf_csv(files: list, write=False) -> pd.DataFrame:
    """
    generate pandas Dataframe structure.
    params:
        TODO
    """
    summary = collections.defaultdict(list)
    for file in files:
        _, density, dataset, model, prune_type, seed, *_ = file.split('/')
        df = pd.read_csv(file)
        df = df.sort_values(by=['epoch'])
        keys = df.keys()
        for k in keys:
            summary[f'{k}'].extend(df[k].tolist())
        summary['prune_type'].extend([prune_type] * len(df.index))

    folder, density, dataset, model, prune_type, seed, *_ = files[0].split('/')
    summary = pd.DataFrame.from_dict(summary)
    name = f"summary-{seed}"
    if write:
        save(summary, osp.join(folder, density, dataset, model, 'csv'), name)
    return summary


if __name__ == "__main__":
    _, path = sys.argv
    prune_type = os.listdir(path)
    graphs = collections.defaultdict(list)
    results = collections.defaultdict(list)
    for p in prune_type:
        if 'csv' in p:
            continue
        subfolder = osp.join(path, p)
        seeds = os.listdir(subfolder)
        for i, seed in enumerate(seeds):
            graphs[seed].append(
                osp.join(subfolder, seed, 'latest', 'graph.pth'))
            results[seed].append(
                osp.join(subfolder, seed, 'latest', 'results.csv'))
    for seed in graphs.keys():
        generate_graph_csv(graphs[seed], True)
        generate_perf_csv(results[seed], True)
