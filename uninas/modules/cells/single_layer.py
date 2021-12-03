"""
basic (search) cells that only contain a single layer
"""

import torch
from uninas.modules.modules.abstract import AbstractModule
from uninas.modules.cells.abstract import AbstractCell, SearchCellInterface, SearchCNNCellInterface
from uninas.modules.primitives.abstract import PrimitiveSet
from uninas.utils.args import Namespace
from uninas.utils.shape import ShapeList
from uninas.register import Register


@Register.network_cell()
class SingleLayerCell(AbstractCell):
    """
    Thinly wraps an operation (layer) as cell
    """
    _num_inputs = 1
    _num_outputs = 1

    def __init__(self, op: AbstractModule, **stored_kwargs):
        super().__init__(**stored_kwargs)
        self._add_to_submodules(op=op)

    def set_dropout_rate(self, p=None) -> int:
        return self.op.set_dropout_rate(p)

    def is_layer(self, cls) -> bool:
        return super().is_layer(cls) or self.op.is_layer(cls)

    def config(self, minimize=False, **_) -> dict:
        if minimize is True:
            return self.op.config(minimize=minimize, **_)
        return super().config(minimize=minimize, **_)

    def _build(self, s_ins: ShapeList, features_mul=1, features_fixed=-1) -> ShapeList:
        assert len(s_ins) == self.num_inputs()
        return ShapeList([self.op.build(s_ins[0], self._num_output_features(s_ins, features_mul, features_fixed))])

    def forward(self, x: [torch.Tensor]) -> [torch.Tensor]:
        """ for '_num_inputs' inputs return '_num_outputs' outputs """
        return [self.op(x[0])]


@Register.network_cell()
class SingleLayerSearchCell(SingleLayerCell, SearchCellInterface):
    """
    Thinly wraps the primitives as cell
    """

    @classmethod
    def search_cell_instance(cls, args: Namespace, index: int, cell_index: int, primitives: PrimitiveSet) -> AbstractCell:
        """
        :param args: global argparse namespace
        :param index: index of this cell
        :param cell_index: index of the cell in the network
        :param primitives: primitives to use in this cell
        :return: search cell
        """
        all_args = cls._all_parsed_arguments(args, index=index)
        arc_key = all_args.pop('arc_key')
        arc_shared = all_args.pop('arc_shared')
        all_args, arc_key = cls._updated_args(all_args, arc_key, arc_shared, cell_index)
        op = primitives.instance(name=arc_key)
        return SingleLayerCell(op=op, **all_args)


@Register.network_cell()
class SingleLayerCNNSearchCell(SingleLayerCell, SearchCNNCellInterface):
    """
    Thinly wraps the primitives as cell
    """

    @classmethod
    def search_cell_instance(cls, args: Namespace, index: int, cell_index: int, primitives: PrimitiveSet) -> AbstractCell:
        """
        :param args: global argparse namespace
        :param index: index of this cell
        :param cell_index: index of the cell in the network
        :param primitives: primitives to use in this cell
        :return: search cell
        """
        all_args = cls._all_parsed_arguments(args, index=index)
        arc_key = all_args.pop('arc_key')
        arc_shared = all_args.pop('arc_shared')
        stride = all_args.pop('stride')
        all_args, arc_key = cls._updated_args(all_args, arc_key, arc_shared, cell_index)
        op = primitives.instance(name=arc_key, stride=stride)
        return SingleLayerCell(op=op, **all_args)


# old classes that are phased out


@Register.network_cell()
class PassThroughCNNCell(SingleLayerCell):
    pass


@Register.network_cell()
class PassThroughCNNSearchCell(SingleLayerCNNSearchCell):
    pass
