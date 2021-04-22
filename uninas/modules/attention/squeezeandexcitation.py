import torch
import torch.nn as nn
from uninas.modules.attention.abstract import AbstractAttentionModule
from uninas.utils.torch.misc import make_divisible
from uninas.register import Register


@Register.attention_module()
class SqueezeExcitationChannelModule(AbstractAttentionModule):
    """
    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
    """

    def __init__(self, c: int, c_substitute: int = None, use_c_substitute=False, divisible: int = None,
                 c_mul=0.25, squeeze_act='relu', excite_act='sigmoid', squeeze_bias=True, excite_bias=True,
                 squeeze_bn=False, squeeze_bn_affine=False):
        """

        :param c: number of input and output channels
        :param c_substitute: used instead of 'c' for calculating inner channels, if not None and 'use_c_substitute'
                             in MobileNet and ShuffleNet blocks this is the number of block input channels
                             (usually fewer than the input channels of the SE module within the block)
        :param use_c_substitute: try using 'c_substitute'
        :param divisible: channels will be a multiple, disabled if None
        :param c_mul: multiplier for inner channels
        :param squeeze_act: activation function after squeezing
        :param excite_act: activation function after exciting
        :param squeeze_bias: use a bias for squeezing
        :param excite_bias: use a bias for exciting
        :param squeeze_bn: use a bn after squeezing
        :param squeeze_bn_affine: use an affine bn
        """
        super().__init__(c, c_substitute, use_c_substitute)
        c_red = make_divisible(int(self.c * c_mul), divisible)
        ops = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c_red, kernel_size=1, stride=1, bias=squeeze_bias)
        ]
        if squeeze_bn:
            ops.append(nn.BatchNorm2d(c_red, affine=squeeze_bn_affine))
        ops.extend([
            Register.act_funs.get(squeeze_act)(inplace=True),
            nn.Conv2d(c_red, c, kernel_size=1, stride=1, bias=excite_bias),
            Register.act_funs.get(excite_act)(inplace=True),
        ])
        self.op = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.op(x)
