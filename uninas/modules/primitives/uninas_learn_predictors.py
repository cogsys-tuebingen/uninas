"""
"""

from uninas.modules.primitives.abstract import CNNPrimitive, PrimitiveSet
from uninas.modules.layers.cnn import ConvLayer
from uninas.modules.layers.scarletnas import LinearTransformerLayer
from uninas.register import Register


@Register.primitive_set()
class UninasLearnPredictorsNoSkip(PrimitiveSet):
    """
    last cell to get the correct size for the head
    """

    @classmethod
    def get_primitives(cls, **primitive_kwargs) -> [CNNPrimitive]:
        df = dict(order='w_bn_act', act_inplace=False, bn_affine=False, use_bn=False, bias=True)
        ops = []
        for act_fun in ['sigmoid', 'tanh', 'relu6', 'swish', 'mish']:
            ops.append(CNNPrimitive(cls=ConvLayer, kwargs=dict(k_size=1, act_fun=act_fun, **df)))
        return ops


@Register.primitive_set()
class UninasLearnPredictorsSkip(PrimitiveSet):
    """
    optional layers, between stem and last cell
    """

    @classmethod
    def get_primitives(cls, **primitive_kwargs) -> [CNNPrimitive]:
        ops = UninasLearnPredictorsNoSkip.get_primitives(**primitive_kwargs)
        ops.append(CNNPrimitive(cls=LinearTransformerLayer, kwargs=dict()))
        return ops
