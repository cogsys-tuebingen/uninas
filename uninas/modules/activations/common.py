"""
common activation functions
"""

import torch.nn as nn
from uninas.register import Register


@Register.act_fun()
def identity(inplace=False) -> nn.Module:
    return nn.Identity()


@Register.act_fun()
def skip(inplace=False) -> nn.Module:
    return nn.Identity()


@Register.act_fun()
def softmax(inplace=False) -> nn.Module:
    return nn.Softmax(dim=-1)


@Register.act_fun()
def relu(inplace=False) -> nn.Module:
    return nn.ReLU(inplace=inplace)


@Register.act_fun()
def relu6(inplace=False) -> nn.Module:
    return nn.ReLU6(inplace=inplace)


@Register.act_fun()
def sigmoid(inplace=False) -> nn.Module:
    return nn.Sigmoid()


@Register.act_fun()
def hsigmoid(inplace=False) -> nn.Module:
    return nn.Hardsigmoid(inplace=inplace)


@Register.act_fun()
def tanh(inplace=False) -> nn.Module:
    return nn.Tanh()


@Register.act_fun()
def htanh(inplace=False) -> nn.Module:
    return nn.Hardtanh(inplace=inplace)
