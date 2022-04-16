import matplotlib.pyplot as plt
import networkx
import numpy as np
import torch
import torch.nn as nn


def mlp_to_network(neuron_network, draw=False):
    """mlp_to_network

    Args:
        neuron_network (numpy.ndarray, or a list of numpy.ndarray): a list of 2D
            matrices that represents the weights in a MLP network. For example,
            we will have weight matrices in shape [(748, 256), (256, 10)] for a
            MLP on MNIST with one hidden layer of dimension 256.
        draw (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    node_list = []
    edge_list = []
    nodes_num = 0
    # nodes_num=nodes_num+(neuron_network.conv.in_channels)*(neuron_network.conv.out_channels)
    # node_list.append(len(node_list)+np.arange(neuron_network.conv.in_channels + neuron_network.conv.out_channels))

    nx_new = networkx.Graph()
    for l, w in enumerate(neuron_network):
        dim_in, dim_out = w.shape
        if l == 0:
            idx_in_start = 0
            idx_out_start = dim_in
        else:
            idx_in_start = idx_out_start
            idx_out_start = idx_out_start + dim_in
        for i in range(dim_in):
            for j in range(dim_out):
                idx_in = idx_in_start + i
                idx_out = idx_out_start + j
                edge_w = np.abs(w[i, j])
                if edge_w > 0:
                    nx_new.add_weighted_edges_from([(idx_in, idx_out, edge_w)])

        if l == 0:
            nodes_num += dim_in + dim_out
            node_list.append(
                list(len(node_list) + np.arange(dim_in + dim_out)))
        else:
            nodes_num += dim_out
            node_list.append(list(len(node_list) + np.arange(dim_out)))

    if draw:
        networkx.draw(nx_new)

        # plt.title('SF Network')
        plt.savefig('sf_base.svg', format='svg')
        plt.savefig('sf_base.png', format='png')
    return nx_new  #, node_list, nodes_num


def conv_to_edges(weight, offset_in=0, offset_out=0, mask=None, mask2=None):
    c_out, c_in, kW, kH = weight.size()
    weight = weight.reshape(c_in, c_out, kW, kH).detach().cpu().numpy()
    mask = mask if mask is not None else [1] * c_in
    assert len(mask) == c_in
    ret = []
    offset_j = offset_in + c_in
    if offset_out > 0:
        offset_j = offset_out

    for i in range(c_in):
        mask_i = mask[i]
        for j in range(c_out):
            if mask2 is not None:
                mask_i *= mask2[j]

            if mask_i:
                ret.append((i + offset_in, offset_j + j, {
                    'weight': weight[i, j].mean((0, 1))
                }))
            else:
                ret.append((i + offset_in, offset_j + j, {'weight': 0.0}))
    return ret, offset_j + c_out


def linear_to_edges(weight, offset=0, mask=None):
    c_out, c_in = weight.size()
    weight = weight.t().detach().cpu().numpy()
    mask = mask if mask is not None else [1] * c_in
    assert len(mask) == c_in

    ret = []

    for i in range(c_in):
        mask_i = mask[i]
        for j in range(c_out):
            if mask_i:
                ret.append((i + offset, offset + c_in + j, {
                    'mask': mask_i,
                    'weight': weight[i, j]
                }))
            else:
                ret.append((i + offset, offset + c_in + j, {
                    'mask': mask_i,
                    'weight': 0.0,
                }))

    return ret, c_in + offset + c_out
