"""
MixConv: Mixed depthwise convolutional kernels
https://arxiv.org/abs/1907.09595
"""

import torch
import torch.nn as nn
from uninas.utils.torch.misc import get_padding, get_splits


class MixConvModule(nn.Module):
    """
    A convolution module with mixed kernel sizes, proposed in MixConv
    """

    def __init__(self, c_in: int, c_out: int, k_size=(3, 5, 7), stride=1, dilation=1, groups=-1, bias=False,
                 padding='same', mode='even', divisible=1):
        super().__init__()
        assert isinstance(k_size, (tuple, list))
        assert c_in == c_out or groups == 1
        self.splits_in = get_splits(c_in, len(k_size), mode=mode, divisible=divisible)
        self.splits_out = get_splits(c_out, len(k_size), mode=mode, divisible=divisible)
        groups = [groups]*len(k_size) if groups > 0 else self.splits_in
        ops = []
        for k, g, si, so in zip(k_size, groups, self.splits_in, self.splits_out):
            p = get_padding(padding, k, stride, dilation)
            ops.append(nn.Conv2d(si, so, k, stride=stride, padding=p, groups=g, bias=bias))
        self.ops = nn.ModuleList(ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits = torch.split(x, self.splits_in, dim=1)
        return torch.cat([op(s) for op, s in zip(self.ops, splits)], dim=1)
