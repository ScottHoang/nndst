import os
import os.path as osp
import sys

import networkx as nx
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

from common_models.models import models
from network_lth.conversion import conv_to_edges
from network_lth.conversion import linear_to_edges
from ticket_scheme.EB import EarlyBird


def stringify(item):
    if isinstance(item, torch.Tensor):
        return str(item.detach().cpu().item())
    else:
        return str(item)


def base_case(module, mask, channel_offset, mask_offset, cnt):
    out_c, in_c = module.weight.shape[0:2]

    if cnt > 0 and mask_offset < mask.size(0):
        if in_c == 512 * 7 * 7:  # vgg hard-coded exception:
            sub_mask = mask[mask_offset:mask_offset + 512]
            sub_mask = sub_mask.tile((49, 1)).t().reshape(-1)
            assert sub_mask.size(0) == in_c
        else:
            sub_mask = mask[mask_offset:mask_offset + in_c]

    else:
        sub_mask = None
    if isinstance(module, nn.Linear):
        edges, new_channel_offset = linear_to_edges(module.weight,
                                                    channel_offset, sub_mask)
    elif isinstance(module, nn.Conv2d):
        edges, new_channel_offset = conv_to_edges(module.weight,
                                                  offset_in=channel_offset,
                                                  mask=sub_mask)
    if cnt > 0 and mask_offset < mask.size(0):
        new_mask_offset = mask_offset + in_c
    else:
        new_mask_offset = mask_offset
    return edges, new_channel_offset, new_mask_offset


def residual_case(module, mask, channel_offset, mask_offset, cnt):
    org_channel_offset, org_mask_offset = channel_offset, mask_offset
    total_edges = []
    out_c, in_c = module.conv1.weight.shape[0:2]
    for name, m in module.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and 'downsample' not in name:
            edges, channel_offset, mask_offset = base_case(
                m, mask, channel_offset, mask_offset, cnt)
            total_edges.extend(edges)
    if module.downsample is not None:
        for name, m in module.downsample.named_modules():
            if isinstance(m, nn.Conv2d):
                if org_mask_offset + in_c < len(
                        mask) and mask_offset + out_c < len(mask):
                    sub_mask1 = mask[org_mask_offset:org_mask_offset + in_c]
                    sub_mask2 = mask[mask_offset:mask_offset + out_c]
                else:
                    sub_mask1 = sub_mask2 = None

                edges, _ = conv_to_edges(m.weight,
                                         offset_in=org_channel_offset,
                                         offset_out=channel_offset - out_c,
                                         mask=sub_mask1,
                                         mask2=sub_mask2)
                mask_offset += out_c
                assert _ == channel_offset

    return total_edges, channel_offset, mask_offset


def main(src_dir, tgt_dir, model_name, num_classes, sparsity, is_eb):
    os.makedirs(tgt_dir, exist_ok=True)
    eb = EarlyBird(sparsity)
    edges = []
    for save_file in os.listdir(src_dir):
        print(save_file)
        if "ckpt" in save_file:
            # print(save_file)
            weights = torch.load(osp.join(src_dir, save_file))['state_dict']
            net = models[model_name](int(num_classes)).cuda()
            net.load_state_dict(weights)
            if is_eb:
                mask = eb.pruning(net, sparsity)
            else:
                mask = torch.tensor([])
            ##############################
            channel_offset = 0
            mask_offset = 0
            total_edges = []
            cnt = 0
            ##############################
            for name, module in net.named_children():
                if isinstance(module, nn.Sequential):
                    for child, child_m in module.named_children():
                        if isinstance(child_m, (BasicBlock, Bottleneck)):
                            # print(child_m)
                            edges, channel_offset, mask_offset = residual_case(
                                child_m, mask, channel_offset, mask_offset,
                                cnt)
                            # print(
                            # f"num edges: {len(edges)}, channel_offset: {channel_offset}, mask_offset: {mask_offset}"
                            # )
                            total_edges.extend(edges)
                            cnt += 1
                        if isinstance(child_m, (nn.Linear, nn.Conv2d)):
                            # print(child_m)
                            edges, channel_offset, mask_offset = base_case(
                                child_m, mask, channel_offset, mask_offset,
                                cnt)
                            # print(
                            # f"num edges: {len(edges)}, channel_offset: {channel_offset}, mask_offset: {mask_offset}"
                            # )
                            total_edges.extend(edges)
                            cnt += 1
                elif isinstance(module, (nn.Conv2d, nn.Linear)):
                    # print(module)
                    edges, channel_offset, mask_offset = base_case(
                        module, mask, channel_offset, mask_offset, cnt)
                    # print(
                    # f"num edges: {len(edges)}, channel_offset: {channel_offset}, mask_offset: {mask_offset}"
                    # )
                    total_edges.extend(edges)
                    cnt += 1
            G = nx.MultiDiGraph()
            G.add_edges_from(total_edges)
            nx.write_gml(G, osp.join(tgt_dir, f"{save_file}.gml"), stringify)
            print(
                f"total nodes: {G.number_of_nodes()}, total edges: {G.number_of_edges()}, save to:{tgt_dir}/{save_file}.gml"
            )


if __name__ == "__main__":
    _, src_dir, tgt_dir, sparsity, is_eb = sys.argv
    is_eb = int(is_eb)
    sparsity = float(sparsity)
    assert osp.isdir(src_dir)
    num_classes = {'cifar10': 10, 'cifar100': 100, 'imagenet': 1000}
    for dataset in os.listdir(src_dir):
        dir_path = osp.join(src_dir, dataset)
        tgt_dir_path = osp.join(tgt_dir, dataset)
        num_class = num_classes[dataset]
        for model_name in os.listdir(dir_path):
            model_path = osp.join(dir_path, model_name)
            tgt_model_path = osp.join(tgt_dir_path, model_name)
            main(model_path, tgt_model_path, model_name, num_class, sparsity,
                 is_eb)
