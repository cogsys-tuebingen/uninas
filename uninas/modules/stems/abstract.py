import torch
from uninas.modules.modules.abstract import AbstractModule, AbstractArgsModule
from uninas.modules.layers.common import SkipLayer
from uninas.utils.args import Argument, Namespace
from uninas.utils.shape import Shape, ShapeList
from uninas.register import Register


class AbstractStem(AbstractArgsModule):
    _num_outputs = None

    @classmethod
    def stem_from_args(cls, args: Namespace) -> AbstractArgsModule:
        """
        :param args: global argparse namespace
        :return: class instance
        """
        return cls.stem_from_kwargs(**cls._all_parsed_arguments(args))

    @classmethod
    def stem_from_kwargs(cls, **kwargs) -> AbstractArgsModule:
        """ get all parsed arguments that the class added """
        raise NotImplementedError

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('features', default=0, type=int, help='num output features of this stem'),
        ]

    @classmethod
    def num_outputs(cls):
        assert cls._num_outputs is not None, "Stem class must define number of outputs"
        return cls._num_outputs

    def set_dropout_rate(self, p=None) -> int:
        n = 0
        for m in self.base_modules(recursive=False):
            n += m.set_dropout_rate(p)
        return n

    def build(self, s_in: Shape) -> ShapeList:
        return super().build(s_in)

    def _build(self, s_in: Shape) -> ShapeList:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> [torch.Tensor]:
        raise NotImplementedError


class SingleLayerStem(AbstractStem):
    """ a single layer as stem """
    _num_outputs = 1

    def __init__(self, stem_module: AbstractModule, **stored_kwargs):
        super().__init__(**stored_kwargs)
        self._add_to_submodules(stem_module=stem_module)

    @classmethod
    def stem_from_kwargs(cls, **kwargs) -> AbstractStem:
        raise NotImplementedError

    def _build(self, s_in: Shape) -> ShapeList:
        """ build the stem for the data set, return list of output feature shapes """
        self.cached['shape_in'] = s_in
        return ShapeList([self.stem_module.build(s_in, self.features)])

    def forward(self, x: torch.Tensor) -> [torch.Tensor]:
        return [self.stem_module(x)]


@Register.network_stem()
class SkipStem(SingleLayerStem):
    """ a skip layer as stem """

    @classmethod
    def stem_from_kwargs(cls, **kwargs) -> AbstractStem:
        return cls(SkipLayer(), **kwargs)
