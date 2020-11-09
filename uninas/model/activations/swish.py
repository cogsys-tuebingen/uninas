import torch
import torch.nn as nn
from uninas.register import Register


class Swish(nn.Module):
    """
    the Swish activation function
    https://arxiv.org/abs/1710.05941
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class PSwish(nn.Module):
    """
    The Swish activation function with a trainable parameter beta
    """

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(size=[1]), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x * self.beta)


@Register.act_fun()
def swish(inplace=False) -> nn.Module:
    return Swish()


@Register.act_fun()
def pswish(inplace=False) -> nn.Module:
    return PSwish()


@Register.act_fun()
def hswish(inplace=False) -> nn.Module:
    return nn.Hardswish()
