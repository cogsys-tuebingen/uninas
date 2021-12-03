"""
basic (search) cells
"""

import torch
from collections.abc import Callable
from functools import partial
from uninas.modules.modules.abstract import AbstractArgsModule
from uninas.modules.primitives.abstract import PrimitiveSet
from uninas.utils.args import ArgsInterface, Argument, Namespace
from uninas.utils.shape import ShapeList


class AbstractCell(AbstractArgsModule):
    _num_inputs = None
    _num_outputs = None

    @classmethod
    def num_inputs(cls):
        assert cls._num_inputs is not None, "Cell class must define number of inputs"
        return cls._num_inputs

    @classmethod
    def num_outputs(cls):
        assert cls._num_outputs is not None, "Cell class must define number of outputs"
        return cls._num_outputs

    def _num_output_features(self, s_ins: ShapeList, features_mul: int, features_fixed: int) -> int:
        if features_fixed > 0:
            nf = features_fixed
        elif self.features_fixed > 0:
            nf = self.features_fixed
        else:
            nf = s_ins[-1].num_features() * self.features_mult
        nf = nf * features_mul
        n, r = divmod(nf, 8)
        if n >= 1:
            return nf-r
        return nf

    @classmethod
    def get_name_in_args(cls, args: Namespace, index=None):
        """ get the name that this block has been assigned in the global argparse args """
        return cls._parsed_argument('name', args, index=index)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('name', type=str, default='c', help='name for the cell order'),
            Argument('features_mult', type=int, default=-1, help='dynamic number of output features'),
            Argument('features_fixed', type=int, default=-1, help='fixed number of output features'),
        ]

    def build(self, s_ins: ShapeList, features_mul=1, features_fixed=-1) -> ShapeList:
        return super().build(s_ins, features_mul=features_mul, features_fixed=features_fixed)

    def _build(self, s_ins: ShapeList, features_mul=1, features_fixed=-1) -> ShapeList:
        """
        :param s_ins: input Shapes
        :param features_mul: global multiplier on number of features
        :param features_fixed: fixed number of output features if >0
        :return:
        """
        raise NotImplementedError

    def forward(self, x: [torch.Tensor]) -> [torch.Tensor]:
        """ for '_num_inputs' inputs return '_num_outputs' outputs """
        raise NotImplementedError


class SearchCellFunctions(ArgsInterface):

    @classmethod
    def _updated_args(cls, all_args: dict, arc_key: str, arc_shared: bool, cell_index: int) -> (dict, str):
        """ update the name to reflect whether the architecture is shared or not """
        if not arc_shared:
            all_args['name'] = '%s-%d' % (all_args['name'], cell_index)
            arc_key = '%s/cell-%d' % (arc_key, cell_index)
        return all_args, arc_key

    @classmethod
    def search_cell_instance(cls, args: Namespace, index: int, cell_index: int, primitives: PrimitiveSet) -> AbstractCell:
        """
        :param args: global argparse namespace
        :param index: index of this cell
        :param cell_index: index of the cell in the network
        :param primitives: primitives to use in this cell
        :return: search cell
        """
        raise NotImplementedError

    @classmethod
    def partial_search_cell_instance(cls, args: Namespace, index: int, primitives: PrimitiveSet) -> Callable:
        """
        :param args: global argparse namespace
        :param index: index of this cell
        :param primitives: primitives to use in this cell
        :return: Callable that requires only the cell_index to complete the search_cell_instance method
        """
        return partial(cls.search_cell_instance, args=args, index=index, primitives=primitives)


class SearchCellInterface(SearchCellFunctions):

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse for when this class is chosen """
        return super().args_to_add(index) + [
            Argument('arc_shared', type=str, default='False', help='whether to use arc_key to share architecture/topology', is_bool=True),
            Argument('arc_key', type=str, default='c', help='key for sharing arc weights'),
        ]

    @classmethod
    def search_cell_instance(cls, args: Namespace, index: int, cell_index: int, primitives: PrimitiveSet) -> AbstractCell:
        """
        :param args: global argparse namespace
        :param index: index of this cell
        :param cell_index: index of the cell in the network
        :param primitives: primitives to use in this cell
        :return: search cell
        """
        raise NotImplementedError


class SearchCNNCellInterface(SearchCellInterface):

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse for when this class is chosen """
        return super().args_to_add(index) + [
            Argument('stride', type=int, default=1, help='stride of this cell'),
        ]

    @classmethod
    def search_cell_instance(cls, args: Namespace, index: int, cell_index: int, primitives: PrimitiveSet) -> AbstractCell:
        """
        :param args: global argparse namespace
        :param index: index of this cell
        :param cell_index: index of the cell in the network
        :param primitives: primitives to use in this cell
        :return: search cell
        """
        raise NotImplementedError


class FixedSearchCellInterface(SearchCellFunctions):
    """
    acts like a search cell, actually fixed. enables searching with fixed structures, like e.g. NAS Bench 201
    """

    @classmethod
    def search_cell_instance(cls, args: Namespace, index: int, cell_index: int, primitives: PrimitiveSet):
        """
        :param args: global argparse namespace
        :param index: index of this cell
        :param cell_index: index of the cell in the network
        :param primitives: primitives to use in this cell
        :return: search cell
        """
        raise NotImplementedError


class FixedSearchCNNCellInterface(FixedSearchCellInterface):

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse for when this class is chosen """
        return super().args_to_add(index) + [
            Argument('stride', type=int, default=1, help='stride of this cell'),
        ]

    @classmethod
    def search_cell_instance(cls, args: Namespace, index: int, cell_index: int, primitives: PrimitiveSet):
        """
        :param args: global argparse namespace
        :param index: index of this cell
        :param cell_index: index of the cell in the network
        :param primitives: primitives to use in this cell
        :return: search cell
        """
        raise NotImplementedError
