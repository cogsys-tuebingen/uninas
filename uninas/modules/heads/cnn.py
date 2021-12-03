import torch
import torch.nn as nn
from uninas.modules.modules.cnn import SqueezeModule
from uninas.modules.heads.abstract import AbstractHead
from uninas.modules.layers.cnn import ClassificationLayer
from uninas.modules.attention.squeezeandexcitation import SqueezeExcitationChannelModule
from uninas.utils.args import Argument
from uninas.utils.shape import Shape
from uninas.register import Register


@Register.network_head()
class ClassificationHead(AbstractHead):
    """
    Network output,
    batchnorm, global average pooling, (dropout), linear
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('bias', default='True', type=str, help='add a bias', is_bool=True),
            Argument('dropout', default=0.0, type=float, help='dropout, <0 to disable entirely'),
        ]

    def set_dropout_rate(self, p=None) -> int:
        return self.head_module.set_dropout_rate(p)

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        self.head_module = ClassificationLayer(bias=self.bias, use_bn=False, use_gap=True, dropout_rate=self.dropout)
        return self.head_module.build(s_in, s_out.num_features())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_module(x)


@Register.network_head()
class FeatureMixClassificationHead(AbstractHead):
    """
    Network output
    conv1x1, global average pooling, linear,
    may also reorder the convolution behind the pooling
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('features', default=1280, type=int, help='num features after the 1x1 convolution'),
            Argument('act_fun', default='relu', type=str, help='act fun of the 1x1 convolution', choices=Register.act_funs.names()),
            Argument('bias', default='True', type=str, help='add a bias', is_bool=True),
            Argument('dropout', default=0.0, type=float, help='initial dropout probability'),
            Argument('gap_first', default='False', type=str, help='GAP before the convolution', is_bool=True),
        ]

    def set_dropout_rate(self, p=None) -> int:
        if p is not None:
            self.head_module[-2].p = p
            return 1
        return 0

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        before, after, squeeze = [], [], [
            nn.AdaptiveAvgPool2d(1),
            SqueezeModule()
        ]
        if self.gap_first:
            after = [
                nn.Linear(s_in.num_features(), self.features, bias=True),  # no affine bn -> use bias
                Register.act_funs.get(self.act_fun)(inplace=True)
            ]
            self.cached['shape_inner'] = Shape([self.features])
        else:
            before = [
                nn.Conv2d(s_in.num_features(), self.features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.features, affine=True),
                Register.act_funs.get(self.act_fun)(inplace=True)
            ]
            self.cached['shape_inner'] = Shape([self.features, s_in.shape[1], s_in.shape[2]])
        ops = before + squeeze + after + [
            nn.Dropout(p=self.dropout),
            nn.Linear(self.features, s_out.num_features(), bias=self.bias)
        ]
        self.head_module = nn.Sequential(*ops)
        return self.probe_outputs(s_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_module(x)


@Register.network_head()
class SeFeatureMixClassificationHead(AbstractHead):
    """
    Network output
    global average pooling, squeeze+excitation, linear, act fun, dropout, linear
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('se_cmul', default=0.25, type=float, help='use Squeeze+Excitation with given width'),
            Argument('se_squeeze_bias', default='True', type=str, help='use SE bias for squeezing', is_bool=True),
            Argument('se_excite_bias', default='True', type=str, help='use SE bias for exciting', is_bool=True),
            Argument('se_act_fun', default='relu', type=str, help='use Squeeze+Excitation with given act fun'),
            Argument('se_bn', default='True', type=str, help='use Squeeze+Excitation with bn', is_bool=True),
            Argument('features', default=1280, type=int, help='num features after the first fc layer'),
            Argument('act_fun', default='relu', type=str, help='act fun of the first fc layer', choices=Register.act_funs.names()),
            Argument('bias0', default='False', type=str, help='add a bias to the first fc layer', is_bool=True),
            Argument('dropout', default=0.0, type=float, help='initial dropout probability'),
            Argument('bias1', default='False', type=str, help='add a bias to the final fc layer', is_bool=True),
        ]

    def set_dropout_rate(self, p=None) -> int:
        if p is not None:
            self.head_module[-2].p = p
            return 1
        return 0

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        ops = [nn.AdaptiveAvgPool2d(1)]
        if self.se_cmul > 0:
            ops.append(SqueezeExcitationChannelModule(s_in.num_features(),
                                                      c_mul=self.se_cmul,
                                                      squeeze_act=self.se_act_fun,
                                                      squeeze_bias=self.se_squeeze_bias and not self.se_bn,
                                                      excite_bias=self.se_excite_bias,
                                                      squeeze_bn=self.se_bn,
                                                      squeeze_bn_affine=self.se_squeeze_bias))
        ops.extend([
            SqueezeModule(),
            nn.Linear(s_in.num_features(), self.features, bias=self.bias0),
            Register.act_funs.get(self.act_fun)(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.features, s_out.num_features(), bias=self.bias1)
        ])
        self.head_module = nn.Sequential(*ops)
        self.cached['shape_inner'] = Shape([self.features])
        return self.probe_outputs(s_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_module(x)
