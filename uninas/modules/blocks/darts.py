"""
blocks as used in the DARTS search space
"""

import torch
import torch.nn as nn
from uninas.methods.strategy_manager import StrategyManager
from uninas.modules.modules.misc import InputChoiceWrapperModule, DropPathModule
from uninas.modules.blocks.abstract import AbstractBlock, SearchCNNBlockInterface
from uninas.modules.layers.common import SkipLayer
from uninas.modules.layers.cnn import ZeroLayer
from uninas.modules.primitives.abstract import PrimitiveSet
from uninas.register import Register
from uninas.utils.shape import Shape, ShapeList


@Register.network_block()
class DartsCNNBlock(AbstractBlock):
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
        results = [m(x) for m in self.ops]
        return sum(results)


@Register.network_block()
class DartsCNNSearchBlock(DartsCNNBlock, SearchCNNBlockInterface):
    """ all mixed ops for one block """

    @classmethod
    def search_block_instance(cls, primitives: PrimitiveSet, arc_key: str, num_inputs: int, stride: int = 1) -> 'DartsCNNSearchBlock':
        """
        :param primitives: primitive set for this block
        :param arc_key: key under which to register architecture parameter names
        :param num_inputs: number of block inputs
        :param stride: stride for the CNN ops
        :return:
        """
        assert num_inputs >= 2
        path_names, ops = [], []

        for i in range(num_inputs):
            s = stride if i < 2 else 1
            path_name = '%s/op-%d' % (arc_key, i)
            wrapped = primitives.instance(name=path_name, stride=s)
            ops.append(InputChoiceWrapperModule(wrapped, idx=i))
            path_names.append(path_name)
        return cls(ops=nn.ModuleList(ops), path_names=path_names)

    def config(self, finalize=True, num_block_ops=2, **_) -> dict:
        """ select the paths with the highest softmax weights, despite them being evaluated separately (like DARTS) """
        if finalize:
            # DARTS style, we enforce having two different inputs for each block, can not model e.g. NASNet
            sm = StrategyManager()
            weights, modules = [], []
            for i, (path_name, op) in enumerate(zip(self.path_names, self.ops)):
                # for each block: remove all paths using the Zero, take the highest weighted remaining one
                w_sm = sm.get_strategy_by_weight(path_name).get_weight_sm(path_name)
                for j, z in enumerate(op.module.wrapped.is_layer_path(ZeroLayer)):
                    if z:
                        w_sm[j].zero_()
                w_sm /= w_sm.sum()
                weights.append((i, w_sm.max()))
            for i, (p_idx, __) in enumerate(sorted(weights, key=lambda w: w[1], reverse=True)):
                # for each block only pick the 'num_block_ops' highest weighted paths now
                if i < num_block_ops:
                    modules.append(self.ops[p_idx])
            return DartsCNNBlock(nn.ModuleList(modules)).config(num_block_ops=num_block_ops, **_)
        else:
            return super().config(finalize=finalize, num_block_ops=num_block_ops, **_)
