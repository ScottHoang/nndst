import torch
import torch.nn as nn

from lib.common_models.models import models
from lib.graphs.scores import pair_layers
from lib.prune.pruning_utils import masked_parameters

if __name__ == "__main__":
    model = models['vgg16'](num_classes=10, seed=1)
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            layers.append(name)
    print(layers)
    print(pair_layers(layers))
