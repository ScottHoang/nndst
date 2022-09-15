import os
import sys

import torch

import common_models as CM
from common_models.utils import init

if __name__ == "__main__":
    "dst: location to save init weights"
    "num_classes: comma seperated number of classes"
    _, seed, dst, num_classes = sys.argv

    torch.manual_seed(int(seed))
    os.makedirs(dst, exist_ok=True)
    num_classes = [int(i) for i in num_classes.split(",")]

    for model_name, fn in CM.models.items():
        for n in num_classes:
            name = f"{model_name}_c-{n}_seed-{seed}.pth.tar"
            path = os.path.join(dst, name)
            model = init(fn(pretrained=False, num_classes=n))
            torch.save(model.state_dict(), path)
