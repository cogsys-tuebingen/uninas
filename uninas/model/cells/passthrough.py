"""
basic (search) cells that only contain a single layer
"""

import torch
from uninas.model.modules.abstract import AbstractModule
from uninas.model.cells.abstract import AbstractCell, SearchCNNCellInterface
from uninas.utils.args import Namespace
from uninas.utils.shape import ShapeList
from uninas.register import Register


@Register.network_cell()
class PassThroughCNNCell(AbstractCell):
    """
    Thinly wraps an operation (layer) as cell
    """
    _num_inputs = 1
    _num_outputs = 1

    def __init__(self, op: AbstractModule, **stored_kwargs):
        super().__init__(**stored_kwargs)
        self._add_to_submodules(op=op)

    def set_dropout_rate(self, p=None):
        self.op.set_dropout_rate(p)

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
class PassThroughCNNSearchCell(PassThroughCNNCell, SearchCNNCellInterface):
    """
    Thinly wraps the primitives as cell
    """

    @classmethod
    def search_cell_instance(cls, args: Namespace, index: int, cell_index: int, strategy_name='default'):
        all_args = cls._all_parsed_arguments(args, index=index)
        arc_key = all_args.pop('arc_key')
        arc_shared = all_args.pop('arc_shared')
        primitives = all_args.pop('primitives')
        stride = all_args.pop('stride')
        all_args, arc_key = cls._updated_args(all_args, arc_key, arc_shared, cell_index)
        op = Register.get(primitives).mixed_instance(name=arc_key, strategy_name=strategy_name, stride=stride)
        return PassThroughCNNCell(op=op, **all_args)
