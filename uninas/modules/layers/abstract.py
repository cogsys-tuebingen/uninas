"""
layers that store info how to save+rebuild them, even with different number of features/channels
"""

import torch
import torch.nn as nn
from uninas.modules.modules.abstract import AbstractModule
from uninas.modules.modules.cnn import PaddingToValueModule
from uninas.utils.shape import Shape
from uninas.register import Register


class AbstractLayer(AbstractModule):

    def build(self, s_in: Shape, c_out: int) -> Shape:
        return super().build(s_in, c_out)

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AbstractStepsLayer(AbstractLayer):
    """
    Basic layer that handles batchnorm, activation function, dropout, and order of those
    they are executed sequentially
    if the layer does not change the number of channels ('changes_c', e.g. pooling), the output is automatically padded
    """
    changes_c = True
    dropout_fun = nn.Dropout2d
    batchnorm_fun = nn.BatchNorm2d
    padding_fun = nn.ConstantPad2d

    def __init__(self, use_bn=True, bn_affine=False, act_fun='identity', act_inplace=False,
                 dropout_rate=0.0, dropout_inplace=False, dropout_keep=False, order='w_bn_act', **__):
        super().__init__(**__)
        self._add_to_kwargs(use_bn=use_bn, bn_affine=bn_affine, act_fun=act_fun, act_inplace=act_inplace,
                            dropout_rate=dropout_rate, dropout_inplace=dropout_inplace, dropout_keep=dropout_keep,
                            order=order)

    def _build(self, s_in: Shape, c_out: int, weight_functions=()) -> Shape:
        steps = []
        for s in self.order.split('_'):
            if s == 'bn' and self.use_bn and self.batchnorm_fun is not None:
                bn = self._get_bn(s_in.num_features(), c_out)
                if bn is not None:
                    steps.append(bn)
            if s == 'w':
                if (self.dropout_rate > 0 or self.dropout_keep) and self.dropout_fun is not None:
                    steps.append(self.dropout_fun(self.dropout_rate, inplace=self.dropout_inplace))
                else:
                    self.dropout_rate = 0.0
                steps.extend(weight_functions)
            if s == 'act':
                act = Register.act_funs.get(self.act_fun)(inplace=self.act_inplace)
                if act is not None:
                    steps.append(act)
        if (c_out > s_in.num_features()) and not self.changes_c:
            steps.append(PaddingToValueModule(c_out, dim=1))
        self.steps = nn.ModuleList(steps)
        return self.probe_outputs(s_in, multiple_outputs=False)

    def _get_bn(self, c_in, c_out):
        """ get bn function with appropriate channel count """
        after_w = False
        for s in self.order.split('_'):
            if s == 'bn':
                return self.batchnorm_fun(c_out if (after_w and self.changes_c) else c_in, affine=self.bn_affine)
            if s == 'w':
                after_w = True

    def first_in_order(self, a: str, b: str):
        for o in self.order.split('_'):
            if o == a:
                return True
            if o == b:
                return False
        return True

    def set_dropout_rate(self, p=None) -> int:
        """ set the dropout rate of every dropout layer to p, no change for p=None. return num of affected modules """
        n = 0
        if self.dropout_fun is not None and isinstance(p, float):
            self.set(dropout_rate=p)
            for s in self.steps:
                if isinstance(s, self.dropout_fun):
                    s.p = p
                    n += 1
        return n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ basic forward, all steps (layer function, dropout, bn) in order """
        for s in self.steps:
            x = s(x)
        return x
