from typing import Union
from uninas.training.result import ResultValue
import torch
import torch.nn as nn


def drop_path(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """ randomly drops a tensor x with a given probability """
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = torch.bernoulli(torch.zeros(size=(x.shape[0], 1, 1, 1), device=x.device)+keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def reset_bn(module: nn.Module):
    """ reset all batch-norm modules """
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()


def count_parameters(model, ignore_aux=True) -> int:
    if ignore_aux:
        return sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'aux' not in n)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_padding(padding: Union[int, str], kernel_size: int, stride: int, dilation: int) -> int:
    """ get padding for either an already fixed number or "same" in given size/stride/dilation settings """
    if isinstance(padding, int):
        return padding
    if padding.lower() == 'same':
        if stride in [1, 2]:
            return (kernel_size * dilation - 1) // 2
        return 0
    raise NotImplementedError


def make_divisible(c: int, divisible: int = None, min_c=None) -> int:
    """ round down number of features, but at most by 10%, optional min_value """
    if divisible is None:
        return c
    min_c = divisible if min_c is None else min_c
    new_c = max(min_c, int(c + divisible / 2) // divisible * divisible)
    if new_c < 0.9 * c:
        new_c += divisible
    return new_c


def get_splits(n: int, n_splits: int, mode='even', divisible=1) -> [int]:
    """
    split the number into 'n_splits' numbers that sum up to 'n' again,
    attempt each number to be divisible by 'divisible'

    mode 'even': all splits should have the same size
    mode 'geo2': each split has only half the size of the previous one
    """
    if mode == 'even':
        splits = [n // n_splits for _ in range(n_splits)]
    elif mode == 'geo2':
        s = [0.5**(i+1) for i in range(n_splits)]
        sm = sum(s)
        splits = [int(n * x / sm) for x in s]
    else:
        raise NotImplementedError
    # pad the first with the remainder
    splits[0] += n - sum(splits)
    # maybe make sure to have sizes divisible be a certain number
    if divisible > 1:
        splits = [make_divisible(s, divisible=divisible) for s in splits]
        splits[0] += n - sum(splits)
    return splits


def itemize(x):
    """ call x.item() on any Tensor in the given dict/list/datatype """
    if isinstance(x, dict):
        return {k: itemize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [itemize(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.item()
    if isinstance(x, ResultValue):
        return x.item()
    return x


def randomize_parameters(module: nn.Module):
    """ set all parameters to normally distributed values """
    for param in module.parameters(recurse=True):
        param.data.zero_()
        param.data.add_(torch.randn(size=param.data.size(), dtype=param.data.dtype, device=param.data.device))
