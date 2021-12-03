import torch
import torch.nn as nn
from uninas.modules.modules.cnn import SqueezeModule
from uninas.modules.heads.abstract import AbstractHead
from uninas.utils.shape import Shape
from uninas.register import Register


@Register.network_head()
class Bench201Head(AbstractHead):
    """
    Network output
    batchnorm, relu, global average pooling, (dropout), linear
    """

    def set_dropout_rate(self, p=None) -> int:
        if p is not None:
            self.head_module[-2].p = p
            return 1
        return 0

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        self.head_module = nn.Sequential(*[
            nn.BatchNorm2d(s_in.num_features()),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            SqueezeModule(),
            nn.Dropout(p=0.0),
            nn.Linear(s_in.num_features(), s_out.num_features(), bias=True)
        ])
        return self.probe_outputs(s_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_module(x)
