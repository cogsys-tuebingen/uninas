import torch
import torch.nn as nn
from typing import Union
from uninas.modules.modules.abstract import AbstractArgsModule, AbstractModule
from uninas.utils.args import Namespace
from uninas.utils.shape import Shape, ShapeList


class AbstractNetworkBody(AbstractArgsModule):

    def __init__(self, **kwargs_to_save):
        super().__init__(**kwargs_to_save)
        self._forward_fun = 0

    def get_head_weightings(self) -> [float]:
        """ get the weights of all heads, in order (the last head at -1 has to be last) """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> [torch.Tensor]:
        """
        returns list of all heads' outputs
        the heads are sorted by ascending cell order
        """
        raise NotImplementedError

    def specific_forward(self, x: Union[torch.Tensor, list], start_cell=-1, end_cell=None) -> [torch.Tensor]:
        """
        can execute specific part of the network,
        returns result after end_block
        """
        raise NotImplementedError

    def build(self, s_in: Shape, s_out: Shape) -> ShapeList:
        return super().build(s_in, s_out)

    def _build(self, s_in: Shape, s_out: Shape) -> ShapeList:
        raise NotImplementedError

    def get_stem(self) -> AbstractModule:
        raise NotImplementedError

    def get_cells(self) -> nn.ModuleList():
        raise NotImplementedError

    def get_heads(self) -> nn.ModuleList():
        raise NotImplementedError

    @classmethod
    def search_network_from_args(cls, args: Namespace, index: int = None, weight_strategies: Union[dict, str] = None):
        """
        :param args: global argparse namespace
        :param index: index of this network
        :param weight_strategies: {strategy name: [cell indices]}, or name used for all, or None for defaults
        """
        raise NotImplementedError
