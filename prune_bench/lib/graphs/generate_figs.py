import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def to_drop(df, drop, axis):
    if drop:
        return df.drop(labels=drop, axis=axis)
    else:
        return df


def savefig(dst, title):
    if dst is not None:
        if title is None:
            title = "anon"
        title = title + ".png"
        plt.savefig(osp.join(dst, title))


def plot(m, drop_c: list = None, dst: str = None, title: str = None):
    m = to_drop(m, drop_c, 1)
    g = sns.FacetGrid(pd.melt(m, 'mask_no'), col="variable", sharey=False)
    g.map(sns.lineplot, 'mask_no', 'value')
    if title is not None:
        g.fig.subplots_adjust(top=0.75)  # adjust the Figure in rp
        g.fig.suptitle(title)
    savefig(dst, title)
    plt.clf()


def plot_layers(m,
                drop_c: list = None,
                drop_r: list = None,
                dst: str = None,
                title: str = None):
    m = to_drop(m, drop_c, 1)
    m = to_drop(m, drop_r, 0)

    m = m.sort_values(by=[
        'mask_no',
        'layer',
    ])
    g = sns.FacetGrid(pd.melt(m, ['mask_no', 'layer']),
                      row="variable",
                      col='layer',
                      sharey='row')
    g.map(sns.lineplot, 'mask_no', 'value')
    if title is not None:
        g.fig.subplots_adjust(top=0.75)  # adjust the Figure in rp
        g.fig.suptitle(title)
    savefig(dst, title)
    plt.clf()


def line_plot(m: pd.DataFrame,
              x: str,
              y: str,
              hue: str,
              dst: str = None,
              title: str = None,
              filter_by: list = None):
    # c = sns.color_palette("flare", as_cmap=True)
    if filter_by:
        masks = None
        for cond in filter_by:
            mask = m.layer.str.contains(cond)
            if masks is None: masks = mask
            else:
                masks |= mask
        m = m[masks]
    sns.set(rc={'figure.figsize': (6.5, 5)})
    g = sns.lineplot(data=m, x=x, y=y, hue=hue)
    if title:
        g.set(title=title)
    savefig(dst, title)
    plt.clf()


if __name__ == "__main__":
    path = "analysis/org_mask_init_train_csv"
    dataset = ['cifar10', 'cifar100']
    variables = [
        'sm', 'rm', 'sparsity', 'sw', 'rw', 't1m', 'copeland_score',
        'compatibility', 'overlap_coefs'
    ]
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    destination = 'analysis/figures'
    for prune_type in os.listdir(path):
        for datatype in dataset:
            dst = osp.join(destination, prune_type, datatype)
            os.makedirs(dst, exist_ok=True)
            p1 = osp.join(path, prune_type, datatype + '.csv')
            p2 = osp.join(path, prune_type, datatype + '_summary.csv')
            df = pd.read_csv(p1)
            for var in variables:
                for layer in layers:
                    line_plot(df,
                              x="mask_no",
                              y=var,
                              hue="layer",
                              filter_by=[layer],
                              dst=dst,
                              title=f'{datatype}_res34_{layer}_{var}')
            df = pd.read_csv(p2)
            df['max_srm'] = df[['sm', 'rm']].max(axis=1)
            plot(df,
                 drop_c=['t1m', 't1w', 'sw', 'rw'],
                 dst=dst,
                 title=f"{datatype}_res34_summary")
