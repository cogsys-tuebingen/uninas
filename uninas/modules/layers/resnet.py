"""
Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385
"""

import torch
import torch.nn as nn
from uninas.modules.layers.abstract import AbstractLayer
from uninas.modules.modules.misc import DropPathModule
from uninas.utils.torch.misc import get_padding
from uninas.utils.shape import Shape
from uninas.register import Register


class AbstractResNetLayer(AbstractLayer):

    def __init__(self, k_size=3, stride=1, padding='same', dilation=1, expansion=1.0,
                 act_fun='relu6', act_inplace=True, has_first_act=False,
                 bn_affine=True, shortcut_type='id'):
        """

        :param k_size: kernel size for the spatial kernel
        :param stride: stride for the both kernels
        :param padding: 'same' or number, for the first kernel
        :param dilation: dilation for the first kernel
        :param expansion: multiplier for inner channels, based on output channels
        :param act_fun: activation function
        :param act_inplace: whether to use the activation function in-place if possible (e.g. ReLU)
        :param has_first_act: whether a the module starts with an activation function
        :param bn_affine: affine batch norm
        :param shortcut_type: shortcut path, [None, 'None', 'id', 'conv1x1', 'avg_conv']
        """
        super().__init__()
        self._add_to_kwargs(k_size=k_size, stride=stride, padding=padding, dilation=dilation, expansion=expansion,
                            act_fun=act_fun, act_inplace=act_inplace, has_first_act=has_first_act,
                            bn_affine=bn_affine, shortcut_type=shortcut_type)
        self._add_to_print_kwargs(has_shortcut=False)
        self.block = None
        self.shortcut = None

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        c_in = s_in.num_features()
        c_inner = int(c_out * self.expansion)
        self.block = self._build_block(c_in, c_inner, c_out, self.has_first_act)
        if self.shortcut_type in [None, 'None']:
            pass
        elif self.shortcut_type == 'id':
            self.shortcut = nn.Identity()
        elif self.shortcut_type == 'conv1x1':
            self.shortcut = nn.Sequential(*[
                nn.Conv2d(c_in, c_out, 1, self.stride, 0, bias=False),
                nn.BatchNorm2d(c_out, affine=self.bn_affine),
            ])
        elif self.shortcut_type == 'avg_conv':
            self.shortcut = nn.Sequential(*[
                nn.AvgPool2d(kernel_size=2, stride=self.stride, padding=0),
                nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False),
            ])
        else:
            raise NotImplementedError('shortcut type "%s" is not implemented' % self.shortcut_type)
        self.has_shortcut = isinstance(self.shortcut, nn.Module)
        if self.has_shortcut:
            self.block = DropPathModule(self.block)
        return self.probe_outputs(s_in)

    def _build_block(self, c_in: int, c_inner: int, c_out: int, has_first_act=False) -> nn.Module:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_shortcut:
            return self.block(x) + self.shortcut(x)
        return self.block(x)


@Register.network_layer()
class ResNetLayer(AbstractResNetLayer):

    def _build_block(self, c_in: int, c_inner: int, c_out: int, has_first_act=False) -> nn.Module:
        padding0 = get_padding(self.padding, self.k_size, self.stride, self.dilation)
        padding1 = get_padding('same', self.k_size, 1, self.dilation)
        ops = [
            nn.Conv2d(c_in, c_inner, self.k_size, self.stride, padding0, self.dilation, bias=False),
            nn.BatchNorm2d(c_inner, affine=self.bn_affine),
            Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            nn.Conv2d(c_inner, c_out, self.k_size, 1, padding1, 1, bias=False),
            nn.BatchNorm2d(c_out, affine=self.bn_affine),
        ]
        if has_first_act:
            return nn.Sequential(Register.act_funs.get(self.act_fun)(inplace=self.act_inplace), *ops)
        return nn.Sequential(*ops)


@Register.network_layer()
class ResNetBottleneckLayer(AbstractResNetLayer):

    def _build_block(self, c_in: int, c_inner: int, c_out: int, has_first_act=False) -> nn.Module:
        padding = get_padding(self.padding, self.k_size, self.stride, self.dilation)
        ops = [
            nn.Conv2d(c_in, c_inner, 1, 1, 0, 1, bias=False),
            nn.BatchNorm2d(c_inner, affine=self.bn_affine),
            Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            nn.Conv2d(c_inner, c_inner, self.k_size, self.stride, padding, self.dilation, bias=False),
            nn.BatchNorm2d(c_inner, affine=self.bn_affine),
            Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            nn.Conv2d(c_inner, c_out, 1, 1, 0, 1, bias=False),
            nn.BatchNorm2d(c_out, affine=self.bn_affine),
        ]
        if has_first_act:
            return nn.Sequential(Register.act_funs.get(self.act_fun)(inplace=self.act_inplace), *ops)
        return nn.Sequential(*ops)
