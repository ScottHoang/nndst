import math
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn


def init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(0.5)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
    return model


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_log_softmax(model):

    def hook(model, inputs, output):
        return nn.functional.log_softmax(output)

    model.register_forward_hook(hook)


def stripping_bias(model: nn.Module, filters: List = None, verbose=True):
    """stripping bias from some layers if filtered or all layers if not filters

    :model: TODO
    :filters: TODO
    :returns: TODO

    """
    for m in model.modules():
        if filters is None or m in filters:
            if hasattr(m, 'bias'):
                del m.bias
                m.register_parameter('bias', None)
                if verbose:
                    print(f"found bias in {m}, removing it ....")
