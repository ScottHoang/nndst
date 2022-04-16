from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn as nn


def add_sparse_args(parser):
    parser.add_argument(
        '--growth',
        type=str,
        default='random',
        help='Growth mode. Choose from: momentum, random, and momentum_neuron.'
    )
    parser.add_argument(
        '--death',
        type=str,
        default='magnitude',
        help=
        'Death mode / pruning mode. Choose from: magnitude, SET, threshold, CS_death.'
    )
    parser.add_argument(
        '--redistribution',
        type=str,
        default='none',
        help=
        'Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.'
    )
    parser.add_argument('--death-rate',
                        type=float,
                        default=0.50,
                        help='The pruning rate / death rate for DST.')
    parser.add_argument('--large-death-rate',
                        type=float,
                        default=0.80,
                        help=' large exploration rate q.')
    parser.add_argument(
        '--PF-rate',
        type=float,
        default=0.8,
        help='The pruning rate / death rate for Pruning and Finetuning.')
    parser.add_argument('--density',
                        type=float,
                        default=0.05,
                        help='The density of the overall sparse network.')
    parser.add_argument('--sparse',
                        action='store_true',
                        help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix',
                        action='store_true',
                        help='Fix topology during training. Default: True.')
    parser.add_argument('--sparse-init',
                        type=str,
                        default='ER',
                        help='sparse initialization')
    parser.add_argument(
        '--update-frequency',
        type=int,
        default=1000,
        metavar='N',
        help='how many iterations to train between parameter exploration')


class Masking(object):

    def __init__(self,
                 death_rate=0.3,
                 growth_death_ratio=1.0,
                 death_rate_decay=None,
                 death_mode='magnitude',
                 growth_mode='gradient',
                 redistribution_mode='none'):
        pass

    def add_module(self, module, density, sparse_init='ER'):
        self.sparse_init = sparse_init
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor,
                                                dtype=torch.float32,
                                                requires_grad=False).cuda()

        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)
        self.init(mode=sparse_init, density=density)

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(
                name, self.masks[name].shape, self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(
                name, self.masks[name + '.weight'].shape,
                self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(
                    name, self.masks[name].shape,
                    np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data * self.masks[name]
                    # if 'momentum_buffer' in self.optimizer.state[tensor]:
                    #     self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]

    def truncate_weights(self, model, pruning_rate):
        print('dynamic sparse training')
        self.modules = [model]
        for module in self.modules:
            for name, weight in module.named_parameters():
                if 'bn' in name or 'bias' in name: continue

                # death
                new_mask = self.magnitude_death(mask, weight, name,
                                                    pruning_rate)

                self.pruning_rate[name] = int(self.masks[name].sum().item() -
                                              new_mask.sum().item())
                self.masks[name][:] = new_mask

        self.apply_mask()
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()

                # growth
                if self.growth_mode == 'random':
                    new_mask = self.random_growth(name, new_mask,
                                                  self.pruning_rate[name],
                                                  weight)

                elif self.growth_mode == 'momentum':
                    new_mask = self.momentum_growth(name, new_mask,
                                                    self.pruning_rate[name],
                                                    weight)

                elif self.growth_mode == 'gradient':
                    new_mask = self.gradient_growth(name, new_mask,
                                                    self.pruning_rate[name],
                                                    weight)

                elif self.growth_mode == 'momentum_neuron':
                    new_mask = self.momentum_neuron_growth(
                        name, new_mask, self.pruning_rate[name], weight)
                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()

        self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters after dst:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1} after dst'.
              format(self.args.density, sparse_size / total_size))

    def pruning(self):
        print('pruning...')
        print('death rate:', self.args.density)
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                num_remove = math.ceil(
                    (1 - self.args.density) * weight.numel())
                x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                self.masks[name].data.view(-1)[idx[:num_remove]] = 0.0
        self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1}'.format(
            self.args.density, sparse_size / total_size))

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.name2death_rate[name] *
                               self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def magnitude_death(self, mask, weight, name, pruning_rate):
        num_zeros = (mask == 0).sum().item()
        num_remove = math.ceil(pruning_rate * (mask.sum().item()))
        if num_remove == 0.0: return weight.data != 0.0
        x, idx = torch.sort(torch.abs(weight.data.view(-1)))

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k - 1].item()

        return (torch.abs(weight.data) > threshold)

    def global_magnitude_death(self):
        death_rate = 0.0
        for name in self.name2death_rate:
            if name in self.masks:
                death_rate = self.name2death_rate[name]
        tokill = math.ceil(death_rate * self.baseline_nonzero)
        total_removed = 0
        prev_removed = 0
        while total_removed < tokill * (1.0 - self.tolerance) or (
                total_removed > tokill * (1.0 + self.tolerance)):
            total_removed = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    remain = (torch.abs(weight.data) >
                              self.threshold).sum().item()
                    total_removed += self.name2nonzeros[name] - remain

            if prev_removed == total_removed: break
            prev_removed = total_removed
            if total_removed > tokill * (1.0 + self.tolerance):
                self.threshold *= 1.0 - self.increment
                self.increment *= 0.99
            elif total_removed < tokill * (1.0 - self.tolerance):
                self.threshold *= 1.0 + self.increment
                self.increment *= 0.99

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.masks[name][:] = torch.abs(weight.data) > self.threshold

        return int(total_removed)

    def global_momentum_growth(self, total_regrowth):
        togrow = total_regrowth
        total_grown = 0
        last_grown = 0
        while total_grown < togrow * (1.0 - self.tolerance) or (
                total_grown > togrow * (1.0 + self.tolerance)):
            total_grown = 0
            total_possible = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    new_mask = self.masks[name]
                    grad = self.get_momentum_for_weight(weight)
                    grad = grad * (new_mask == 0).float()
                    possible = (grad != 0.0).sum().item()
                    total_possible += possible
                    grown = (torch.abs(grad.data) >
                             self.growth_threshold).sum().item()
                    total_grown += grown
            print(total_grown, self.growth_threshold, togrow,
                  self.growth_increment, total_possible)
            if total_grown == last_grown: break
            last_grown = total_grown

            if total_grown > togrow * (1.0 + self.tolerance):
                self.growth_threshold *= 1.02
                #self.growth_increment *= 0.95
            elif total_grown < togrow * (1.0 - self.tolerance):
                self.growth_threshold *= 0.98
                #self.growth_increment *= 0.95

        total_new_nonzeros = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                new_mask = self.masks[name]
                grad = self.get_momentum_for_weight(weight)
                grad = grad * (new_mask == 0).float()
                self.masks[name][:] = (
                    new_mask.byte() |
                    (torch.abs(grad.data) > self.growth_threshold)).float()
                total_new_nonzeros += new_mask.sum().item()
        return total_new_nonzeros

    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.name2death_rate[name] *
                               self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove / 2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k - 1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove / 2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k - 1].item()

        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)

        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def random_growth(self, name, new_mask, total_regrowth, weight):
        n = (new_mask == 0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth / n)
        new_weights = torch.rand(
            new_mask.shape).cuda() < expeced_growth_probability  #lsw
        # new_weights = torch.rand(new_mask.shape) < expeced_growth_probability
        return new_mask.byte() | new_weights

    def momentum_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)
        grad = grad * (new_mask == 0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, total_regrowth, weight):
        if self.density_dict[name] == 1.0:
            new_mask = torch.ones_like(new_mask,
                                       dtype=torch.float32,
                                       requires_grad=False).cuda()
        else:
            grad = self.get_gradient_for_weights(weight)
            grad = grad * (new_mask == 0).float()

            y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
            new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def momentum_neuron_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask == 0).sum(sum_dim)

        M = M * (new_mask == 0).float()
        for i, fraction in enumerate(v):
            neuron_regrowth = math.floor(fraction.item() * total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''

    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1 / (torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                if name in self.name2variance:
                    val = '{0}: {1}->{2}, density: {3:.3f}, proportion: {4:.4f}'.format(
                        name, self.name2nonzeros[name], num_nonzeros,
                        num_nonzeros / float(mask.numel()),
                        self.name2variance[name])
                    print(val)
                else:
                    print(name, num_nonzeros)

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.death_rate))
                break

    def reset_momentum(self):
        """
        Taken from: https://github.com/AlliedToasters/synapses/blob/master/synapses/SET_layer.py
        Resets buffers from memory according to passed indices.
        When connections are reset, parameters should be treated
        as freshly initialized.
        """
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                weights = list(self.optimizer.state[tensor])
                for w in weights:
                    if w == 'momentum_buffer':
                        # momentum
                        self.optimizer.state[tensor][w][
                            mask == 0] = torch.mean(
                                self.optimizer.state[tensor][w][mask.byte()])
                        # self.optimizer.state[tensor][w][mask==0] = 0
                    elif w == 'square_avg' or \
                        w == 'exp_avg' or \
                        w == 'exp_avg_sq' or \
                        w == 'exp_inf':
                        # Adam
                        self.optimizer.state[tensor][w][
                            mask == 0] = torch.mean(
                                self.optimizer.state[tensor][w][mask.byte()])

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte(
                ) & self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(
                    self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(
                    self.fired_masks[name].sum().item()) / float(
                        self.fired_masks[name].numel())
                print('Layerwise percentage of the fired weights of', name,
                      'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights / ntotal_weights
        print('The percentage of the total fired weights is:',
              total_fired_weights)
        return layer_fired_weights, total_fired_weights
