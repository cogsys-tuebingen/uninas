"""
common interface to internal and external networks
"""


from typing import Union
import torch
import torch.nn as nn
from uninas.models.networks.abstract2 import Abstract2Network
from uninas.modules.modules.abstract import AbstractModule
from uninas.modules.networks.abstract import AbstractNetworkBody
from uninas.utils.args import Namespace, Argument
from uninas.utils.shape import Shape, ShapeList


class AbstractUninasNetwork(Abstract2Network):

    def __init__(self, net: Union[AbstractNetworkBody, None], log_detail=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_to_submodules(net=net)
        self._add_to_kwargs(log_detail=log_detail)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('log_detail', default=5, type=int, help='how detailed to log the network, smaller = less'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> 'AbstractUninasNetwork':
        """
        :param args: global argparse namespace
        :param index: argument index
        """
        raise NotImplementedError

    def _build2(self, s_in: Shape, s_out: Shape) -> ShapeList:
        """ build the network """
        return self.net.build(s_in, s_out)

    def _set_dropout_rate(self, p=None) -> int:
        """ set the dropout rate of every dropout layer to p """
        return self.net.set_dropout_rate(p=p)

    def get_head_weightings(self) -> [float]:
        """ get the weights of all heads, in order """
        return self.get_network().get_head_weightings()

    def get_network(self) -> AbstractNetworkBody:
        return self.net

    def get_stem(self) -> AbstractModule:
        return self.get_network().get_stem()

    def get_cells(self) -> nn.ModuleList():
        return self.get_network().get_cells()

    def get_heads(self) -> nn.ModuleList():
        return self.get_network().get_heads()

    def _get_cell_input_shapes(self) -> ShapeList:
        return ShapeList([c.get_shape_in() for c in self.get_cells()])

    def _get_cell_output_shapes(self) -> ShapeList:
        return ShapeList([c.get_shape_out() for c in self.get_cells()])

    def all_forward(self, x: torch.Tensor) -> [torch.Tensor]:
        """
        returns list of all heads' outputs
        the heads are sorted by ascending cell order
        """
        return self.net.forward(x)

    def specific_forward(self, x: Union[torch.Tensor, list], start_cell=-1, end_cell=None) -> [torch.Tensor]:
        """
        can execute specific part of the network,
        returns result after end_cell
        """
        return self.net.specific_forward(x, start_cell=start_cell, end_cell=end_cell)

    def str(self, depth=0, **_) -> str:
        return super().str(depth=depth, max_depth=self.log_detail, **_)
