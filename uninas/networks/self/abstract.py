"""
common interface to internal and external networks
"""


from typing import Union
import torch.nn as nn
from uninas.networks.abstract2 import Abstract2Network
from uninas.model.modules.abstract import AbstractModule
from uninas.model.networks.abstract import AbstractNetworkBody
from uninas.utils.args import Namespace, Argument
from uninas.utils.shape import Shape, ShapeList


class AbstractUninasNetwork(Abstract2Network):

    def __init__(self, name: str, net: AbstractNetworkBody, checkpoint_path: str, log_detail=5):
        super().__init__(name, checkpoint_path)
        self._add_to_submodules(net=net)
        self._add_to_kwargs(log_detail=log_detail)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('log_detail', default=5, type=int, help='how detailed to log the network, smaller = less'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None, weight_strategies: Union[dict, str] = "default"):
        """
        :param args: global argparse namespace
        :param index: argument index
        :param weight_strategies: {strategy name: [cell indices]}, or name used for all
        """
        raise NotImplementedError

    def _build2(self, s_in: Shape, s_out: Shape) -> Shape:
        return self.net.build(s_in, s_out)

    def _set_dropout_rate(self, p=None):
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

    def _get_input_shapes(self) -> ShapeList:
        return ShapeList([c.get_cached('shape_in') for c in self.get_cells()])

    def _get_output_shapes(self) -> ShapeList:
        return ShapeList([c.get_cached('shape_out') for c in self.get_cells()])

    def forward(self, *args, **kwargs):
        return self.net.forward(*args, **kwargs)

    def str(self, depth=0, **_) -> str:
        return super().str(depth=depth, max_depth=self.log_detail, **_)
