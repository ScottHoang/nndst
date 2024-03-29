from __future__ import print_function

import argparse
import collections
import copy
import hashlib
import json
import logging
import os
import time
import warnings

import pandas as pd
import sparselearning
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from sparselearning.common_models.models import models as MODELS
from sparselearning.common_models.utils import add_log_softmax
from sparselearning.core import CosineDecay
from sparselearning.core import Masking
from sparselearning.utils import get_cifar100_dataloaders
from sparselearning.utils import get_cifar10_dataloaders
from sparselearning.utils import get_mnist_dataloaders
from sparselearning.utils import plot_class_feature_histograms

warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}
for name, fn in MODELS.items():
    models[name] = (fn, [])


def merge_mask(model, mask_path, start_from_init=False):
    assert os.path.isfile(mask_path)
    if start_from_init:
        random_weight = model.state_dict()
        masked_weight = torch.load(mask_path, map_location='cpu')
        final_mask = {}
        for k in masked_weight.keys():
            rw = random_weight[k].cpu()
            rm = (masked_weight[k] != 0.0).float()
            final_mask[k] = rw * rm
    else:
        final_mask = torch.load(mask_path, map_location='cpu')
    model.load_state_dict(final_mask, strict=False)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            mask = (module.weight != 0).float()
            if isinstance(module, nn.BatchNorm2d):
                prune.CustomFromMask.apply(module, 'bias', mask)
            prune.CustomFromMask.apply(module, 'weight', mask)
    return model


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("SAVING")
    torch.save(state, filename)


def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(
        args.model, args.density,
        hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s',
                                  datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


def train(args, model, device, train_loader, optimizer, epoch, mask=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        pred = output.argmax(
            dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None: mask.step()
        else: optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '
                .format(epoch, batch_idx * len(data),
                        len(train_loader) * args.batch_size,
                        100. * batch_idx / len(train_loader), loss.item(),
                        correct, n, 100. * correct / float(n)))

    # training summary
    print_and_log(
        '\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            'Training summary', train_loss / batch_idx, correct, n,
            100. * correct / float(n)))


def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            # model.t = target
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log(
        '\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            'Test evaluation' if is_test_set else 'Evaluation', test_loss,
            correct, n, 100. * correct / float(n)))
    return correct / float(n)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier',
                        type=int,
                        default=1,
                        metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs',
                        type=int,
                        default=250,
                        metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.1,
                        metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=17,
                        metavar='S',
                        help='random seed (default: 17)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='sgd',
        help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5e-4)
    parser.add_argument(
        '--iters',
        type=int,
        default=1,
        help=
        'How many times the model should be run after each other. Default=1')
    parser.add_argument(
        '--save-features',
        action='store_true',
        help=
        'Resumes a saved model and saves its feature data to disk for plotting.'
    )
    parser.add_argument(
        '--bench',
        action='store_true',
        help='Enables the benchmarking of layers and estimates sparse speedups'
    )
    parser.add_argument('--max-threads',
                        type=int,
                        default=10,
                        help='How many threads to use for data loading.')
    parser.add_argument(
        '--decay-schedule',
        type=str,
        default='cosine',
        help=
        'The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.'
    )
    parser.add_argument('--nolrsche',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--no_rewire_extend',
                        action='store_true',
                        default=False,
                        help='if ture, only do rewire for 250 epoochs')
    parser.add_argument('-j',
                        '--workers',
                        default=10,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--world-size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--mgpu',
                        action='store_true',
                        help='Enable snip initialization. Default: True.')
    parser.add_argument("--mask_path", type=str, default='')
    parser.add_argument("--start_from_init", "-sfi", action='store_true')
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)
    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'device will be chosen as {device} for this run.')

    print_and_log('\n\n')
    print_and_log('=' * 80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))

        if args.data == 'mnist':
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(
                args, validation_split=args.valid_split)
            outputs = 10
        elif args.data == 'cifar10':
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(
                args, args.valid_split, max_threads=args.max_threads)
            outputs = 10
        elif args.data == 'cifar100':
            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(
                args, args.valid_split, max_threads=args.max_threads)
            outputs = 100
        if args.model not in models:
            print(
                'You need to select an existing model via the --model argument. Available models include: '
            )
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        else:
            cls, cls_args = models[args.model]
            model = cls(num_classes=outputs).cuda()
            add_log_softmax(model)
            print_and_log(model)
            print_and_log('=' * 60)
            print_and_log(args.model)
            print_and_log('=' * 60)

            print_and_log('=' * 60)
            print_and_log('Prune mode: {0}'.format(args.death))
            print_and_log('Growth mode: {0}'.format(args.growth))
            print_and_log('Redistribution mode: {0}'.format(
                args.redistribution))
            print_and_log('=' * 60)

        if args.mgpu:
            print('Using multi gpus')
            model = torch.nn.DataParallel(model).to(device)

        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.l2)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.lr,
                                   weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        if args.nolrsche:
            lr_scheduler = None
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(args.epochs / 2),
                            int(args.epochs * 3 / 4)],
                last_epoch=-1)
        if args.resume:
            if os.path.isfile(args.resume):
                print_and_log("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print_and_log("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']))
                print_and_log('Testing...')
                evaluate(args, model, device, test_loader)
                model.feats = []
                model.densities = []
                plot_class_feature_histograms(args, model, device,
                                              train_loader, optimizer)
            else:
                print_and_log("=> no checkpoint found at '{}'".format(
                    args.resume))

        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale=None,
                                       dynamic_loss_scale=True,
                                       dynamic_loss_args={'init_scale': 2**16})
            model = model.half()

        timestr = args.mask_path.split("/")[-3]
        mask_type = args.mask_path.split("/")[-2]
        mask_no = args.mask_path.split("/")[-1]
        branching = 'branches' if not args.start_from_init else "from_init"
        save_dir = os.path.join('results', args.data, 'performance', timestr,
                                mask_type, branching)
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        model = merge_mask(model, args.mask_path, args.start_from_init)

        best_acc = 0.0
        acc = collections.defaultdict(list)
        kill_count = cnt = 20
        for epoch in range(1, args.epochs * args.multiplier + 1):

            t0 = time.time()
            train(args, model, device, train_loader, optimizer, epoch, None)

            if lr_scheduler: lr_scheduler.step()

            val_acc = evaluate(args, model, device, valid_loader)
            acc['val_acc'].append(val_acc)
            acc['epoch'].append(epoch)
            if best_acc < val_acc:
                cnt = kill_count
                best_acc = val_acc
            else:
                cnt -= 1

            print_and_log(
                'Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'
                .format(optimizer.param_groups[0]['lr'],
                        time.time() - t0))
            if cnt == 0:
                print("Early Breaking")
                break

        df = pd.DataFrame.from_dict(acc)
        df.to_csv(os.path.join(save_dir, mask_no + ".csv"))

        print_and_log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))


if __name__ == '__main__':
    main()
