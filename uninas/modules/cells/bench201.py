"""
cells as used in the NAS Bench 201 search space
"""

import torch
import torch.nn as nn
from uninas.modules.cells.abstract import AbstractCell, SearchCNNCellInterface, FixedSearchCNNCellInterface
from uninas.modules.cells.single_layer import SingleLayerCell
from uninas.modules.layers.resnet import ResNetLayer
from uninas.modules.primitives.abstract import PrimitiveSet
from uninas.register import Register
from uninas.utils.args import Argument, Namespace
from uninas.utils.shape import ShapeList


@Register.network_cell()
class Bench201CNNCell(AbstractCell):
    """
    A cell like in the NAS-Bench-201 / NATS-Bench setup.
    Each block receives the cell input and the outputs of all prior blocks as input,
    the output of the final block is used as cell output.
    """
    _num_inputs = 1
    _num_outputs = 1

    def __init__(self, blocks: nn.ModuleList, **stored_kwargs):
        super().__init__(**stored_kwargs)
        self._add_to_submodule_lists(blocks=blocks)

    def _build(self, s_ins: ShapeList, features_mul=1, features_fixed=-1) -> ShapeList:
        assert len(s_ins) == self.num_inputs()
        c_out = self._num_output_features(s_ins, features_mul, features_fixed)

        inner_shapes = s_ins.copy()
        for m in self.blocks:
            s = m.build(inner_shapes, c_out)
            inner_shapes.append(s)
        return ShapeList([s_ins[-1]])

    def forward(self, x: [torch.Tensor]) -> [torch.Tensor]:
        """ for '_num_inputs' inputs return '_num_outputs' outputs """
        for m in self.blocks:
            x.append(m(x))
        return [x[-1]]


@Register.network_cell()
class Bench201CNNSearchCell(Bench201CNNCell, SearchCNNCellInterface):
    """
    A search cell like in the NAS-Bench-201 / NATS-Bench setup.
    Each block receives the cell input and the outputs of all prior blocks as input,
    the output of the final block is used as cell output.
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('num_blocks', default=3, type=int, help='num blocks in the cell'),
            Argument('cls_block', default='Bench201CNNSearchBlock', type=str, help='class to use as block', choices=Register.network_blocks.names()),
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
        all_args = cls._all_parsed_arguments(args, index=index)
        arc_key = all_args.pop('arc_key')
        arc_shared = all_args.pop('arc_shared')
        stride = all_args.pop('stride')
        num_blocks = all_args.pop('num_blocks')
        cls_block = all_args.pop('cls_block')
        all_args, arc_key = cls._updated_args(all_args, arc_key, arc_shared, cell_index)

        cls_block = Register.network_blocks.get(cls_block)

        blocks = []
        for i in range(num_blocks):
            num_inputs = i + cls._num_inputs
            block_arc_key = '%s/block-%d/%d' % (arc_key, i, num_inputs)
            block = cls_block.search_block_instance(primitives, arc_key=block_arc_key, num_inputs=num_inputs, stride=stride)
            blocks.append(block)

        return Bench201CNNCell(nn.ModuleList(blocks), **all_args)


@Register.network_cell()
class Bench201ReductionCell(AbstractCell, FixedSearchCNNCellInterface):
    """
    A ResNet reduction block with expansion 1; and average pooling with a linear 1x1 Convolution as shortcut
    """

    @classmethod
    def search_cell_instance(cls, args: Namespace, index: int, cell_index: int, primitives: PrimitiveSet) -> SingleLayerCell:
        """
        :param args: global argparse namespace
        :param index: index of this cell
        :param cell_index: index of the cell in the network
        :param primitives: primitives to use in this cell
        :return: search cell
        """
        all_args = cls._all_parsed_arguments(args, index=index)
        stride = all_args.pop('stride')
        all_args, arc_key = cls._updated_args(all_args, 'r', False, cell_index)
        op = ResNetLayer(k_size=3, stride=stride, expansion=1.0,
                         act_fun='relu', act_inplace=False,  has_first_act=True, shortcut_type='avg_conv')
        return SingleLayerCell(op=op, **all_args)
