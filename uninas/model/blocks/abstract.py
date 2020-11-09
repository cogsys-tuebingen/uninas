"""
basic (search) blocks
"""

import torch
from uninas.utils.args import ArgsInterface, Namespace
from uninas.model.modules.abstract import AbstractArgsModule
from uninas.utils.shape import Shape, ShapeList


class AbstractBlock(AbstractArgsModule):

    def _build(self, s_ins: ShapeList, num_features: int) -> Shape:
        raise NotImplementedError

    def forward(self, x: [torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class SearchBlockInterface(ArgsInterface):

    @classmethod
    def get_name_in_args(cls, args: Namespace, index=None):
        return cls._parsed_argument('name', args, index=index)

    @classmethod
    def search_block_instance(cls, primitives: str, arc_key: str, num_inputs: int, strategy_name='default'):
        """
        :param primitives: primitive set for this block
        :param arc_key: key under which to register architecture parameter names
        :param num_inputs: number of block inputs
        :param strategy_name: name of the strategy object to register the search parameters with
        :return:
        """
        raise NotImplementedError


class SearchCNNBlockInterface(SearchBlockInterface):

    @classmethod
    def search_block_instance(cls, primitives: str, arc_key: str, num_inputs: int, strategy_name='default', stride: int = 1):
        """
        :param primitives: primitive set for this block
        :param arc_key: key under which to register architecture parameter names
        :param num_inputs: number of block inputs
        :param strategy_name: name of the strategy object to register the search parameters with
        :param stride:
        :return:
        """
        raise NotImplementedError
