"""
cells as used in the DARTS search space
"""

import torch
import torch.nn as nn
from uninas.modules.modules.misc import ConcatChoiceModule
from uninas.modules.cells.abstract import AbstractCell, SearchCNNCellInterface
from uninas.modules.layers.cnn import FactorizedReductionLayer, ConvLayer
from uninas.modules.primitives.abstract import PrimitiveSet
from uninas.register import Register
from uninas.utils.args import Argument, Namespace
from uninas.utils.shape import ShapeList


@Register.network_cell()
class DartsCNNCell(AbstractCell):
    """
    A cell like in DARTS.
    Each block receives the cell inputs and the outputs of all prior blocks as input,
    the block outputs are concatenated and used as cell output.
    """
    _num_inputs = 2     # fixed to exactly 2 right now
    _num_outputs = 2

    def __init__(self, blocks: nn.ModuleList, concat: ConcatChoiceModule, **stored_kwargs):
        super().__init__(**stored_kwargs)
        self._add_to_submodule_lists(blocks=blocks)
        self._add_to_submodules(concat=concat)
        self.preprocess = nn.ModuleList()

    def _build(self, s_ins: ShapeList, features_mul=1, features_fixed=-1) -> ShapeList:
        assert len(s_ins) == self.num_inputs()
        c_out = self._num_output_features(s_ins, features_mul, features_fixed)
        c_inner = c_out // self.concat.num
        is_prev_reduce = s_ins[0].shape[1] > s_ins[1].shape[1]
        base_kwargs = dict(use_bn=True, bn_affine=True, act_fun='relu', order='act_w_bn')
        # if the previous layer reduces the spatial size, the layer before that has larger sizes than this one!
        if is_prev_reduce:
            self.preprocess.append(FactorizedReductionLayer(stride=2, **base_kwargs))
        else:
            self.preprocess.append(ConvLayer(k_size=1, dilation=1, stride=1, **base_kwargs))
        s_inner_p0 = self.preprocess[0].build(s_ins[0], c_inner)
        self.preprocess.append(ConvLayer(k_size=1, dilation=1, stride=1, **base_kwargs))
        s_inner_p1 = self.preprocess[1].build(s_ins[1], c_inner)
        inner_shapes = [s_inner_p0, s_inner_p1]
        for m in self.blocks:
            s = m.build(inner_shapes, c_inner)
            inner_shapes.append(s)
        s_ins.append(self.concat.build(inner_shapes, c_out))
        return ShapeList(s_ins[-self._num_outputs:])

    def forward(self, x: [torch.Tensor]) -> [torch.Tensor]:
        """ for '_num_inputs' inputs return '_num_outputs' outputs """
        states = [p(xi) for p, xi in zip(self.preprocess, x)]
        for m in self.blocks:
            states.append(m(states))
        x.append(self.concat(states))
        return x[-self._num_outputs:]


@Register.network_cell()
class DartsCNNSearchCell(DartsCNNCell, SearchCNNCellInterface):
    """
    A cell like in DARTS.
    Each block receives the cell inputs and the outputs of all prior blocks as input,
    the block outputs are concatenated and used as cell output.
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('num_concat', default=4, type=int, help='num blocks concatenated as cell output'),
            Argument('num_blocks', default=4, type=int, help='num blocks in the cell'),
            Argument('cls_block', default='DartsCNNSearchBlock', type=str, help='class to use as block', choices=Register.network_blocks.names()),
        ]

    @classmethod
    def search_cell_instance(cls, args: Namespace, index: int, cell_index: int, primitives: PrimitiveSet) -> DartsCNNCell:
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
        num_concat = all_args.pop('num_concat')
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
        concat_output = list(range(len(blocks)+cls._num_inputs))[-num_concat:]
        concat = ConcatChoiceModule(idxs=concat_output, dim=1)

        return DartsCNNCell(nn.ModuleList(blocks), concat, **all_args)
