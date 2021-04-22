"""
blocks as used in the NAS Bench 201 search space
"""

import torch
import torch.nn as nn
from uninas.modules.modules.misc import InputChoiceWrapperModule, DropPathModule
from uninas.modules.blocks.abstract import AbstractBlock, SearchCNNBlockInterface
from uninas.modules.layers.common import SkipLayer
from uninas.modules.primitives.abstract import PrimitiveSet
from uninas.register import Register
from uninas.utils.shape import Shape, ShapeList


@Register.network_block()
class Bench201CNNBlock(AbstractBlock):
    def __init__(self, ops: nn.ModuleList, **stored_kwargs):
        super().__init__(**stored_kwargs)
        ops = nn.ModuleList([DropPathModule(op, op.is_layer(SkipLayer), drop_p=0.0, drop_ids=True) for op in ops])
        self._add_to_submodule_lists(ops=ops)

    def _build(self, s_ins: ShapeList, c_out: int) -> Shape:
        shapes = [m.build(s_ins, c_out) for m in self.ops]
        for s0, s1 in zip(shapes[:-1], shapes[1:]):
            assert s0 == s1
        return shapes[-1].copy()

    def forward(self, x: [torch.Tensor]) -> torch.Tensor:
        return sum([m(x) for m in self.ops])


@Register.network_block()
class Bench201CNNSearchBlock(Bench201CNNBlock, SearchCNNBlockInterface):
    """ all mixed ops for one block """

    @classmethod
    def search_block_instance(cls, primitives: PrimitiveSet, arc_key: str, num_inputs: int, stride: int = 1) -> 'Bench201CNNSearchBlock':
        """
        :param primitives: primitive set for this block
        :param arc_key: key under which to register architecture parameter names
        :param num_inputs: number of block inputs
        :param stride: stride for the CNN ops
        :return:
        """
        assert num_inputs >= 1
        path_names, ops = [], []

        for i in range(num_inputs):
            s = stride if i < 2 else 1
            path_name = '%s/op-%d' % (arc_key, i)
            wrapped = primitives.instance(name=path_name, stride=s)
            ops.append(InputChoiceWrapperModule(wrapped, idx=i))
            path_names.append(path_name)
        return cls(ops=nn.ModuleList(ops), path_names=path_names)
