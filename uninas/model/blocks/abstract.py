"""
basic (search) blocks
"""

import torch
from uninas.utils.args import ArgsInterface, Namespace
from uninas.model.modules.abstract import AbstractArgsModule
from uninas.model.primitives.abstract import PrimitiveSet
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
    def search_block_instance(cls, primitives: PrimitiveSet, arc_key: str, num_inputs: int) -> 'SearchBlockInterface':
        """
        :param primitives: primitive set for this block
        :param arc_key: key under which to register architecture parameter names
        :param num_inputs: number of block inputs
        :return:
        """
        raise NotImplementedError


class SearchCNNBlockInterface(SearchBlockInterface):

    @classmethod
    def search_block_instance(cls, primitives: PrimitiveSet, arc_key: str, num_inputs: int, stride: int = 1) -> 'SearchCNNBlockInterface':
        """
        :param primitives: primitive set for this block
        :param arc_key: key under which to register architecture parameter names
        :param num_inputs: number of block inputs
        :param stride: stride for the CNN ops
        :return:
        """
        raise NotImplementedError
