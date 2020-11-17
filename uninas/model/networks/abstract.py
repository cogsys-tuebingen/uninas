import types
import torch
import torch.nn as nn
from typing import Union
from uninas.model.modules.abstract import AbstractArgsModule, AbstractModule
from uninas.utils.args import Namespace
from uninas.utils.shape import Shape


class AbstractNetworkBody(AbstractArgsModule):

    def get_head_weightings(self) -> [float]:
        """ get the weights of all heads, in order (the last head at -1 has to be last) """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> [torch.Tensor]:
        raise NotImplementedError

    def forward_specific(self, x: Union[torch.Tensor, list], start_block=-1, end_block=None) -> [torch.Tensor]:
        raise NotImplementedError

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        raise NotImplementedError

    def get_stem(self) -> AbstractModule:
        raise NotImplementedError

    def get_cells(self) -> nn.ModuleList():
        raise NotImplementedError

    def get_heads(self) -> nn.ModuleList():
        raise NotImplementedError

    def use_specific_forward(self):
        """
        replace the default forward pass with one that can execute specific cells, can not be undone
        """
        assert len(self.heads) == 1, 'Can not have multiple heads!'
        self.forward = types.MethodType(self.forward_specific, self)

    @classmethod
    def search_network_from_args(cls, args: Namespace, weight_strategies: Union[dict, str]):
        """
        :param args: global argparse namespace
        :param weight_strategies: {strategy name: [cell indices]}, or name used for all
        """
        raise NotImplementedError
