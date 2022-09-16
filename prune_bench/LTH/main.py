# Importing Libraries import argparse
import collections
import copy
import json
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.prune as prune
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from common_models import models
from common_models.utils import set_seed
from tqdm import tqdm

import utils

model = None
# Custom Libraries

# Tensorboard initialization

# Plotting Style
sns.set_style('darkgrid')


# Main
def main(args, ITE=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.prune_type == "reinit" else False
    set_seed(args.seed)

    # Data Loader
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    if args.data == "cifar10":
        traindataset = datasets.CIFAR10('../data',
                                        train=True,
                                        download=True,
                                        transform=transform)
        testdataset = datasets.CIFAR10('../data',
                                       train=False,
                                       transform=transform)
        num_classes = 10

    elif args.data == "cifar100":
        traindataset = datasets.CIFAR100('../data',
                                         train=True,
                                         download=True,
                                         transform=transform)
        testdataset = datasets.CIFAR100('../data',
                                        train=False,
                                        transform=transform)
        num_classes = 100

    # If you want to add extra datasets paste here

    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader = torch.utils.data.DataLoader(traindataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=False)
    #train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(testdataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              drop_last=True)

    # Importing Network Architecture
    global model
    model = models[args.arch_type](num_classes=num_classes, seed=args.seed)
    # for m in model.modules():
    # if isinstance(m, (nn.Linear, nn.Conv2d)):
    # if hasattr(m, 'bias'):
    # print(f"found bias in {m} removing it....")
    # del m.bias
    # m.register_parameter("bias", None)
    model = model.cuda()
    timestr = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    save_dir = os.path.join(args.result_dir, f"density_{args.density}",
                            args.data, args.arch_type, 'lth', str(args.seed),
                            timestr)
    os.makedirs(save_dir, exist_ok=True)

    # Weight Initialization
    # model.apply(weight_init)

    # Making Initial Mask
    make_mask(model)
    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    torch.save(initial_state_dict, os.path.join(save_dir, 'init.pth.tar'))

    # Optimizer and Loss
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.end_iter / 2),
                    int(args.end_iter * 3 / 4)],
        last_epoch=-1)
    criterion = nn.CrossEntropyLoss()  # Default was F.nll_loss

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    step = 0
    all_loss = np.zeros(args.end_iter, float)
    all_accuracy = np.zeros(args.end_iter, float)

    with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    for _ite in range(args.start_iter, ITERATION):
        if not _ite == 0:
            prune_by_percentile(args.prune_percent,
                                resample=resample,
                                reinit=False)
            original_initialization(initial_state_dict)
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=args.lr,
                                        momentum=0.9,
                                        weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(args.epochs / 2),
                            int(args.epochs * 3 / 4)],
                last_epoch=-1)
        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter), total=args.end_iter)

        df = collections.defaultdict(list)
        model_density = check_density(model)
        save_dir = os.path.join(args.result_dir,
                                f"density_{model_density:.2f}", args.data,
                                args.arch_type, 'lth', str(args.seed), timestr)
        os.makedirs(save_dir, exist_ok=True)
        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(
                        {
                            'state_dict': model.state_dict(),
                            'epoch': iter_ + 1
                        }, os.path.join(save_dir, 'model.pth'))

            # Training
            loss = train(model, train_loader, optimizer, criterion)

            df['loss'].append(loss)
            df['val_acc'].append(accuracy)
            df['test_acc'].append(accuracy)
            df['epoch'].append(iter_)

            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%'
                )
            lr_scheduler.step()
        df = pd.DataFrame.from_dict(df)
        df.to_csv(os.path.join(save_dir, "results.csv"))


def get_prune_models(model):
    model = copy.deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            mask = (module.weight != 0.0).float()
            prune.CustomFromMask.apply(module, 'weight', mask)
    return model.state_dict()


# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()
        optimizer.step()
    return train_loss.item()


# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(
                1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def check_density(model):

    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if hasattr(m, 'weight_mask'):
            sum_list = sum_list + float(m.weight_orig.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight_mask == 0))
    print('* remain weight = ', 100 * (1 - zero_sum / sum_list), '%')

    return 1 - zero_sum / sum_list


# Prune by Percentile module
def prune_by_percentile(percent, resample=False, reinit=False, **kwargs):

    global model

    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig'):
            param = module.weight_orig * module.weight_mask
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(
                tensor)]  # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            weight_dev = param.device
            new_mask = np.where(
                abs(tensor) < percentile_value, 0,
                module.weight_mask.cpu().numpy())
            new_mask = torch.tensor(new_mask).to(weight_dev)

            # Apply new weight and mask
            module.weight_mask.data.copy_(new_mask)


# Function to make an empty mask of the same size as the model


def make_mask(model):
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            masks = torch.ones_like(module.weight)
            prune.CustomFromMask.apply(module, 'weight', masks)


def original_initialization(initial_state_dict):
    global model

    for name, param in model.named_parameters():
        if 'weight_orig' in name:
            param.data.copy_(initial_state_dict[name])
        if 'bias' in name:
            param.data.copy_(initial_state_dict[name])


# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def get_iter_by_density(density, prune_perc) -> float:
    import math
    prune_perc = 1.0 - (prune_perc / 100.0)
    num_iter = math.log(density) / math.log(prune_perc)

    return int(math.floor(num_iter))


if __name__ == "__main__":

    #from gooey import Gooey
    #@Gooey

    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=250, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type",
                        default="lt",
                        type=str,
                        help="lt | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--data",
                        default="mnist",
                        type=str,
                        help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument(
        "--arch_type",
        default="fc1",
        type=str,
    )
    parser.add_argument("--prune_percent",
                        default=20,
                        type=int,
                        help="Pruning percent")
    parser.add_argument("--prune_iterations",
                        default=35,
                        type=int,
                        help="Pruning iterations count")
    parser.add_argument("--seed", default=1, type=int)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    resample = False

    args.prune_iterations = get_iter_by_density(args.density,
                                                args.prune_percent)
    main(args, ITE=1)
