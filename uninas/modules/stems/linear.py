import torch
from uninas.modules.stems.abstract import AbstractStem
from uninas.modules.modules.abstract import AbstractModule
from uninas.modules.layers.common import LinearLayer
from uninas.utils.args import Argument
from uninas.utils.shape import Shape, ShapeList
from uninas.utils.misc import split
from uninas.register import Register


@Register.network_stem()
class LinearToConvStem(AbstractStem):
    """ changes 1D (C) to 3D (Cx1x1) data, which enables convolutions """
    _num_outputs = 1

    def __init__(self, stem: AbstractModule, **stored_kwargs):
        super().__init__(**stored_kwargs)
        self._add_to_submodules(stem=stem)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('act_fun', default='relu6', type=str, help='act fun of the conv layer', choices=Register.act_funs.names()),
            Argument('use_bias', default='True', type=str, help='use a bias', is_bool=True),
            Argument('use_bn', default='True', type=str, help='use batchnorm', is_bool=True),
        ]

    @classmethod
    def stem_from_kwargs(cls, **kwargs) -> AbstractStem:
        stem = LinearLayer(act_fun=kwargs.get('act_fun'), bias=kwargs.get('use_bias'), use_bn=kwargs.get('use_bn'))
        return cls(stem, **kwargs)

    def _build(self, s_in: Shape) -> ShapeList:
        """ build the stem for the data set, return list of output feature sizes """
        self.stem.build(s_in, self.features)
        return self.probe_outputs(s_in, multiple_outputs=True)

    def forward(self, x: torch.Tensor) -> [torch.Tensor]:
        return [self.stem(x).unsqueeze(-1).unsqueeze(-1)]


@Register.network_stem()
class ReshapeStem(AbstractStem):
    """ simply reshapes data """
    _num_outputs = 1

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('reshape_to', default='1, 1, -1', type=str, help='new shape of the data'),
        ]

    @classmethod
    def stem_from_kwargs(cls, **kwargs) -> AbstractStem:
        """ get all parsed arguments that the class added """
        kwargs['reshape_to'] = split(kwargs['reshape_to'], int)
        return cls(**kwargs)

    def _build(self, s_in: Shape) -> ShapeList:
        """ build the stem for the data set, return list of output feature sizes """
        assert (self.features <= -1) or (self.features == s_in.numel()),\
            "num features (%d) must be <0 or match the number of input features (%d)" % (self.features, s_in.numel())
        return self.probe_outputs(s_in, multiple_outputs=True)

    def forward(self, x: torch.Tensor) -> [torch.Tensor]:
        return [torch.reshape(x, [x.shape[0]] + self.reshape_to)]
