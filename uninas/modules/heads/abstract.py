import torch
from uninas.modules.modules.abstract import AbstractArgsModule
from uninas.utils.args import Argument, Namespace
from uninas.utils.shape import Shape


class AbstractHead(AbstractArgsModule):

    def config(self, **_) -> dict:
        if self.persist:
            return super().config(**_)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('weight', default=1, type=float, help='loss multiplier for this head'),
            Argument('cell_idx', default=-1, type=int, help='at which cell index to add this head'),
            Argument('persist', default='True', type=str, help='add this head to the network config', is_bool=True),
        ]

    @classmethod
    def head_from_args(cls, args: Namespace, index=None) -> AbstractArgsModule:
        """
        :param args: global argparse namespace
        :param index: head index in the global argparse namespace
        :return: class instance
        """
        return cls.head_from_kwargs(**cls._all_parsed_arguments(args, index=index))

    @classmethod
    def head_from_kwargs(cls, **kwargs) -> AbstractArgsModule:
        return cls(**kwargs)

    def set_dropout_rate(self, p=None) -> int:
        raise NotImplementedError

    def build(self, s_in: Shape, s_out: Shape) -> Shape:
        return super().build(s_in, s_out)

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
