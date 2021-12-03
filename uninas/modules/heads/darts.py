import torch
from uninas.modules.heads.abstract import AbstractHead
from uninas.modules.modules.abstract import AbstractModule
from uninas.utils.shape import Shape
from uninas.register import Register
import torch.nn as nn


class BasicDartsAuxHeadModule(nn.Module):
    def __init__(self, c: int, num_classes: int, init_pool_stride: int):
        super(BasicDartsAuxHeadModule, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=init_pool_stride, padding=0, count_include_pad=False),
            nn.Conv2d(c, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class BasicDartsAuxHead(AbstractModule):
    def __init__(self, init_pool_stride: int):
        super().__init__()
        self.auxiliary = None
        self._add_to_kwargs(init_pool_stride=init_pool_stride)

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        self.auxiliary = BasicDartsAuxHeadModule(c=s_in.num_features(), num_classes=c_out,
                                                 init_pool_stride=self.init_pool_stride)
        return self.probe_outputs(s_in, multiple_outputs=False)

    def forward(self, x):
        return self.auxiliary(x)


@Register.network_head()
class DartsCifarAuxHead(AbstractHead):
    """
    CIFAR network auxiliary head as in DARTS
    """

    def set_dropout_rate(self, p=None) -> int:
        return self.head_module.set_dropout_rate(p)

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        """ assuming input size 8x8 """
        self.head_module = BasicDartsAuxHead(init_pool_stride=3)
        return self.head_module.build(s_in, s_out.num_features())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_module(x)


@Register.network_head()
class DartsImageNetAuxHead(DartsCifarAuxHead):
    """
    ImageNet network auxiliary head as in DARTS
    """

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        """ assuming input size 14x14 """
        self.head_module = BasicDartsAuxHead(init_pool_stride=2)
        return self.head_module.build(s_in, s_out.num_features())
