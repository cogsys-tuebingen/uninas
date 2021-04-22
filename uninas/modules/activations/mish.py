import torch
import torch.nn as nn
import torch.nn.functional as F
from uninas.register import Register


@torch.jit.script
def mish_fun(x: torch.Tensor):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    see https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/functional.py
    """
    return x * torch.tanh(F.softplus(x))


class Mish(nn.Module):
    """
    the Mish activation function
    https://arxiv.org/abs/1908.08681
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mish_fun(x)


@Register.act_fun()
def mish(inplace=False) -> nn.Module:
    return Mish()
