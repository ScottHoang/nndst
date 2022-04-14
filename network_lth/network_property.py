
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import dataset
import matplotlib.pyplot as plt
import matplotlib 

import torchvision
import torchvision.transforms as transforms
import numpy as np
import networkx 
import models.cifar_resnet as resnet20
import json

import shutil, os
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 on ResNet20 analysis')

parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt_path='/mnt/guihong/lth/open_lth_data/lottery_93bc65d66dfa64ffaf2a0ab105433a2c/replicate_1/level_'
net=resnet20.Model.get_model_from_name('cifar_resnet_20')

# class simple_net(nn.Module):
def max_filter_value(input_data):
    data_size=input_data.size()
    temp_data= torch.abs(torch.transpose(input_data, 0, 1))
    temp_data=temp_data.reshape(data_size[1],-1)    
    return torch.max(temp_data,dim=1,keepdim=False)

def mean_filter_value(input_data):
    data_size=input_data.size()
    temp_data= torch.abs(torch.transpose(input_data, 0, 1))
    temp_data=temp_data.reshape(data_size[1],-1)    
    return torch.mean(temp_data,dim=1,keepdim=False)

def conv_layer_filter_network(func,input_data,layer, net,node_in_idx,node_out_idx,batch_idx,rules='avg_filter_value'):
    groups=layer.groups
    out=layer(input_data)
    in_channels=layer.in_channels
    out_channels=layer.out_channels
    batch_size=input_data.size()[0]
    # input_array=np.array(input_data.detch().item())# B C W H
    
    single_channel   = int(in_channels/groups)
    out_side_channel = int(out_channels/groups)
    in_filter_value  = torch.zeros(groups, single_channel, int(out_side_channel))
    out_filter_value = torch.zeros(groups, single_channel, int(out_side_channel))
    wgt_filter_value = torch.zeros(groups, single_channel, int(out_side_channel))


    for i in range(groups):
        temp_data = func(input_data[:,i*single_channel:(i+1)*single_channel,:,:])

        in_filter_value[i,:,:]  = temp_data.unsqueeze(dim=1).repeat(1,int(out_side_channel))
        for j in range(single_channel):
            temp_out=F.conv2d(input_data[:,i*single_channel+j:i*single_channel+j+1,:,:],layer.weight[i*int(out_side_channel):(i+1)*out_side_channel,j:j+1,:,:])
            out_filter_value[i,j,:]=func(temp_out)

    in_filter_value=in_filter_value.view(-1)
    out_filter_value=out_filter_value.view(-1)

    wgt_filter_value=wgt_filter_value.view(-1)
    for i in range(wgt_filter_value.size()[0]):
        if in_filter_value[i]!=0:
            wgt_filter_value[i]=out_filter_value[i]/in_filter_value[i]
        else:
            wgt_filter_value[i]=0

    wgt_filter_value=wgt_filter_value.view(groups, single_channel, int(out_side_channel))

    edge_array=np.zeros((in_channels,out_channels))
    for i in range(groups):
        # print(edge_array[i*single_channel:(i+1)*single_channel,i*out_side_channel:(i+1)*out_side_channel])
        # print((wgt_filter_value[i,:,:].detach().cpu().item()))
        # print(wgt_filter_value[0,:,:])

        edge_array[i*single_channel:(i+1)*single_channel,i*out_side_channel:(i+1)*out_side_channel]=np.array(wgt_filter_value[i,:,:].detach().cpu())
    
    # print(edge_array)
    in_node_list=np.arange(in_channels)+node_in_idx
    out_node_list=np.arange(out_channels)+node_out_idx

    for i in range(in_channels):
        for j in range(out_channels):
            if batch_idx==0:
                # print(in_node_list[i],out_node_list[j],edge_array[i][j])
                net.add_weighted_edges_from([(int(in_node_list[i]),int(out_node_list[j]),edge_array[i][j])])
            else:
                # print(batch_idx)
                net.edges[in_node_list[i],out_node_list[j]]['weight']=net.edges[in_node_list[i],out_node_list[j]]['weight']+edge_array[i][j]
                # print(in_node_list[i],out_node_list[j],net.edges[in_node_list[i],out_node_list[j]]['weight'])
                if batch_idx==99:
                    net.edges[in_node_list[i],out_node_list[j]]['weight']=net.edges[in_node_list[i],out_node_list[j]]['weight']/100.0

    return net

def add_short_cut_link(net,node_in_idx,node_out_idx,num_channels):
    in_node_list=np.arange(num_channels)+node_in_idx
    out_node_list=np.arange(num_channels)+node_out_idx
    for i in range(num_channels):
        # print(i+node_in_idx,i+node_out_idx,1)
        net.add_weighted_edges_from([(i+node_in_idx,i+node_out_idx,1)])
    return net
'''conv2d +9 block [1,1,1,3-x,1,1,6-x,1,1]'''
def test_net(given_model,test_loader):
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    batch_num = 0
    given_model.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            #print(inputs.size())
            outputs = given_model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_num = batch_idx
    print('Test  Loss: %.6f | Acc: %.3f%%  | (%d/%d)'
          % (test_loss/(batch_num+1), 100.*correct/total, correct, total))


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
                edge_w = np.abs(w[i,j])
                if edge_w > 0:
                    nx_new.add_weighted_edges_from([(idx_in, idx_out, edge_w)])
        
        if l == 0:
            nodes_num += dim_in + dim_out
            node_list.append(list(len(node_list) + np.arange(dim_in + dim_out)))
        else:
            nodes_num += dim_out
            node_list.append(list(len(node_list) + np.arange(dim_out)))

    if draw:
        networkx.draw(nx_new)

        # plt.title('SF Network')
        plt.savefig('sf_base.svg',format='svg')
        plt.savefig('sf_base.png',format='png')
    return nx_new  #, node_list, nodes_num


def net_filter_network(neuron_network,testloader):
    node_list=[]
    edge_list=[]
    nodes_num=0
    nodes_num=nodes_num+(neuron_network.conv.in_channels)*(neuron_network.conv.out_channels)
    node_list.append(len(node_list)+np.arange(neuron_network.conv.in_channels+neuron_network.conv.out_channels))

    nx_new=networkx.Graph()
    neuron_network=neuron_network.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            node_in_idx=0
            node_out_idx=neuron_network.conv.in_channels

            inputs, targets = inputs.to(device), targets.to(device)
            nx_new=conv_layer_filter_network(func=mean_filter_value,input_data=inputs,layer=neuron_network.conv,net=nx_new,
                                node_in_idx=node_in_idx,node_out_idx=node_out_idx,batch_idx=batch_idx)
            out=F.relu(neuron_network.bn(neuron_network.conv(inputs)))

            node_in_idx=node_out_idx#+neuron_network.conv.out_channels
            for block in neuron_network.blocks:
                if len(block.shortcut)>0:
                    node_out_idx=node_in_idx+block.conv1.in_channels+block.conv1.out_channels
                    nx_new=conv_layer_filter_network(func=mean_filter_value,input_data=out,layer=block.shortcut[0],
                        net=nx_new,node_in_idx=node_in_idx,node_out_idx=node_out_idx,batch_idx=batch_idx)
                elif batch_idx==0:
                    node_out_idx=node_in_idx+block.conv1.in_channels+block.conv1.out_channels
                    nx_new=add_short_cut_link(net=nx_new,node_in_idx=node_in_idx,
                            node_out_idx=node_out_idx,num_channels= block.conv1.in_channels)
                out_temp=out 

                node_out_idx=node_in_idx+block.conv1.in_channels
                nx_new=conv_layer_filter_network(func=mean_filter_value,input_data=out,layer=block.conv1,net=nx_new,
                                node_in_idx=node_in_idx,node_out_idx=node_out_idx,batch_idx=batch_idx)

                node_in_idx=node_out_idx#+block.conv1.out_channels 
                node_out_idx=node_in_idx+block.conv2.in_channels               
                out=F.relu(block.bn1(block.conv1(out)))
                nx_new=conv_layer_filter_network(func=mean_filter_value,input_data=out,layer=block.conv2,net=nx_new,
                                node_in_idx=node_in_idx,node_out_idx=node_out_idx,batch_idx=batch_idx)
                
                out=block.bn2(block.conv2(out))+block.shortcut(out_temp)
                node_in_idx=node_out_idx#+block.conv2.out_channels 
    


    networkx.draw(nx_new)

    # plt.title('SF Network')
    plt.savefig('sf_base.svg',format='svg')
    plt.savefig('sf_base.png',format='png')
    return nx_new

if args.dataset=='cifar10':
    print('using cifar-10 dataset')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
elif args.dataset=='cifar100':
    print('using cifar-100 dataset')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)


ckpt_path='/mnt/guihong/lth/open_lth_data/lottery_93bc65d66dfa64ffaf2a0ab105433a2c/replicate_1/level_'
def resume_ckpt(ckpt_name,net_name='cifar_resnet_20',wm=1,level_idx=0):
    net=resnet20.Model.get_model_from_name(net_name)


    checkpoint = torch.load(ckpt_name)

    net.load_state_dict(checkpoint)
    return net#,start_epoch,best_acc

def scan_ckpt(sparsity_list,num_ckpt=21):
    for i in range(num_ckpt):
        ckpt_ep160_name=ckpt_path+str(i)+'/main/model_ep160_it0.pth'
        ckpt_net=resume_ckpt(ckpt_ep160_name)
        test_net(ckpt_net,testloader)
        # nx_net=net_filter_network(ckpt_net,testloader)
        # networkx.write_gexf(nx_net, 'nexf_logs/ep160_{}_sparsity_{:.4f}.gexf'.format(i,1-sparsity_list[i]))
    for i in range(num_ckpt):
        ckpt_ep0_name=ckpt_path+str(i)+'/main/model_ep0_it0.pth'
        # ckpt_net=resume_ckpt(ckpt_ep0_name)
        # nx_net=net_filter_network(ckpt_net,testloader)
        # networkx.write_gexf(nx_net, 'nexf_logs/ep0_{}_sparsity_{:.4f}.gexf'.format(i,1-sparsity_list[i]))
    for i in range(num_ckpt):
        mask_name=ckpt_path+str(i)+'/main/mask.pth'
        # ckpt_net=resume_ckpt(mask_name)
        # nx_net=net_filter_network(ckpt_net,testloader)
        # networkx.write_gexf(nx_net, 'nexf_logs/mask_{}_sparsity_{:.4f}.gexf'.format(i,1-sparsity_list[i]))

def scan_one_ckpt(level_idx=0):
    ckpt_ep0_name=ckpt_path+str(level_idx)+'/main/model_ep0_it0.pth'
    ckpt_ep160_name=ckpt_path+str(level_idx)+'/main/model_ep160_it0.pth'
    mask_name=ckpt_path+str(level_idx)+'/main/mask.pth'
    ckpt_net=resume_ckpt(mask_name)
    net_filter_network(ckpt_net,testloader)    
    # net_filter_network(neuron_network,testloader)
# scan_one_ckpt()

def scan_json(num_ckpt=21):
    data=[]
    for i in range(num_ckpt):
        json_name=ckpt_path+str(i)+'/main/sparsity_report.json'
        with open(json_name) as f:
            data.append( json.load(f))
    # print(data)
    sparsity_list=np.array([float (float(i['unpruned'])/270896) for i in data])
    print(sparsity_list)
    return sparsity_list
def prune_model_perc(model, perc):
    for name, param in model.state_dict().items():
        if("conv" in name and "weight" in name and "bn" not in name):
            param_cpu = param.detach().cpu().numpy()
            thres = np.percentile(np.abs(param_cpu), perc * 100)
            param[param.abs() < thres] = 0
        if("fc" in name and "weight" in name and "bn" not in name):
            param_cpu = param.detach().cpu().numpy()
            thres = np.percentile(np.abs(param_cpu), perc * 100)
            param[param.abs() < thres] = 0
    return model
def scan_sparsity(sparsity_list):

    ckpt_ep160_name=ckpt_path+str(0)+'/main/model_ep160_it0.pth'
    ckpt_ep160=resume_ckpt(ckpt_ep160_name)
    ckpt_ep0_name=ckpt_path+str(0)+'/main/model_ep0_it0.pth'
    ckpt_ep0=resume_ckpt(ckpt_ep0_name)
        
    for idx, sparsity in enumerate(sparsity_list):
        # net_160=deepcopy(ckpt_ep160)
        # net_160=prune_model_perc(model=net_160,perc=1-sparsity)
        # nx_net=net_filter_network(net_160,testloader)
        # # test_net(net_160,testloader)
        # networkx.write_gexf(nx_net, 'nexf_logs/pruned_{}_ep160_sparsity_{:.4f}.gexf'.format(idx,1-sparsity))

        net_0=deepcopy(ckpt_ep0)
        net_0=prune_model_perc(model=net_0,perc=1-sparsity)
        nx_net=net_filter_network(net_0,testloader)
        networkx.write_gexf(nx_net, 'nexf_logs/pruned_{}_ep0_sparsity_{:.4f}.gexf'.format(idx,1-sparsity))


sparsity_list=scan_json()
scan_sparsity(sparsity_list)
# scan_ckpt(sparsity_list)
# scan_one_ckpt()
