import torch
import torchvision

__all__ = ['resnet18', 'resnet34', 'resnet50', 'vgg16', 'models']

import os

path1 = "./common_models/random_weights"
path2 = "../common_models/random_weights"

if os.path.isdir(path1):
    RANDOM_PATH = path1
else:
    RANDOM_PATH = path2


def resnet18(*args, **kwargs):
    n = resolve(args, kwargs)
    model = torchvision.models.resnet18(num_classes=n)
    try_load_random(model, os.path.join(RANDOM_PATH, f"resnet18_{n}.pth.tar"))
    return model


def resnet34(*args, **kwargs):
    n = resolve(args, kwargs)
    model = torchvision.models.resnet34(num_classes=n)  #*args, **kwargs)
    try_load_random(model, os.path.join(RANDOM_PATH, f"resnet34_{n}.pth.tar"))
    return model


def resnet50(*args, **kwargs):
    n = resolve(args, kwargs)
    model = torchvision.models.resnet50(num_classes=n)  #*args, **kwargs)
    try_load_random(model, os.path.join(RANDOM_PATH, f"resnet50_{n}.pth.tar"))
    return model


def vgg16(*args, **kwargs):
    n = resolve(args, kwargs)
    model = torchvision.models.vgg16_bn(num_classes=n)  #*args, **kwargs)
    try_load_random(model, os.path.join(RANDOM_PATH, f"vgg16_{n}.pth.tar"))
    return model


def resolve(args, kwargs):
    n = kwargs.get("num_classes", None)
    if n is None:
        dataset = kwargs.get("dataset", None)
        if dataset is None:
            if len(args):
                n = args[-1]
            else:
                n = 1000
        else:
            if dataset.lower() == 'cifar10':
                n = 10
            elif dataset.lower() == 'cifar100':
                n = 100
            elif dataset.lower() == 'imagenet':
                n = 1000
    return n


def try_load_random(model, path):
    if os.path.isfile(path):
        print("load random weights")
        model.load_state_dict(torch.load(path))
    else:
        print("fail to load random weights")


models = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "vgg16": vgg16
}
