import torch
from uninas.modules.modules.abstract import AbstractModule
from uninas.modules.stems.abstract import AbstractStem
from uninas.utils.shape import Shape, ShapeList
from uninas.register import Register
import uninas.modules.layers.cnn as layers


@Register.network_stem()
class DartsCifarStem(AbstractStem):
    """ standard stem of CIFAR DARTS models, a single ConvLayer, returned twice """
    _num_outputs = 2

    def __init__(self, stem_module: AbstractModule, **stored_kwargs):
        super().__init__(**stored_kwargs)
        self._add_to_submodules(stem_module=stem_module)

    @classmethod
    def stem_from_kwargs(cls, **kwargs) -> AbstractStem:
        stem_module = layers.ConvLayer(k_size=3, dilation=1, stride=1, act_fun=None, dropout_rate=0.0, order='w_bn',
                                       use_bn=True, bn_affine=True)
        return cls(stem_module, **kwargs)

    def _build(self, s_in: Shape) -> ShapeList:
        s = self.stem_module.build(s_in, self.features)
        return ShapeList([s, s])

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        r = self.stem_module(x)
        return [r, r]


@Register.network_stem()
class DartsImagenetStem(AbstractStem):
    """ standard stem of Imagenet DARTS models, three stacked ConvLayers, returning the last two """
    _num_outputs = 2

    def __init__(self, stem00: AbstractModule, stem01: AbstractModule, stem1: AbstractModule, **stored_kwargs):
        super().__init__(**stored_kwargs)
        self._add_to_submodules(stem00=stem00, stem01=stem01, stem1=stem1)

    @classmethod
    def stem_from_kwargs(cls, **kwargs) -> AbstractStem:
        shared = dict(use_bn=True, bn_affine=True, bias=False, padding=1, stride=2, k_size=3)
        stem00 = layers.ConvLayer(act_fun='relu', order='w_bn_act', **shared)
        stem01 = layers.ConvLayer(act_fun=None, order='w_bn', **shared)
        stem1 = layers.ConvLayer(act_fun='relu', order='act_w_bn', **shared)
        return cls(stem00, stem01, stem1, **kwargs)

    def _build(self, s_in: Shape) -> ShapeList:
        cm = self.features // 2
        s0 = self.stem00.build(s_in, cm)
        s1 = self.stem01.build(s0, self.features)
        s2 = self.stem1.build(s1, self.features)
        return ShapeList([s1, s2])

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        r0 = self.stem01(self.stem00(x))
        r1 = self.stem1(r0)
        return [r0, r1]
