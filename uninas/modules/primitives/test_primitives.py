"""
just testing some things
"""

from uninas.modules.primitives.abstract import CNNPrimitive, PrimitiveSet
from uninas.modules.layers.shufflenet import ShuffleNetV2Layer, ShuffleNetV2XceptionLayer
from uninas.register import Register


@Register.primitive_set()
class Test0Primitives(PrimitiveSet):
    """
    many ShuffleNetV2Xception blocks,
    just to check whether the GPU loads all paths (out of memory) or just one (should be fine)
    """

    @classmethod
    def get_primitives(cls, **primitive_kwargs) -> [CNNPrimitive]:
        df = dict(dilation=1, act_inplace=False, bn_affine=True, use_bn=True, act_fun='relu', expansion=1.0)
        return [
            CNNPrimitive(cls=ShuffleNetV2Layer, kwargs=dict(k_size=3, dilation=1, expansion=1.0, act_fun='relu')),
            CNNPrimitive(cls=ShuffleNetV2XceptionLayer, kwargs=dict(k_size=7, **df)),
            CNNPrimitive(cls=ShuffleNetV2XceptionLayer, kwargs=dict(k_size=7, **df)),
            CNNPrimitive(cls=ShuffleNetV2XceptionLayer, kwargs=dict(k_size=7, **df)),
            CNNPrimitive(cls=ShuffleNetV2XceptionLayer, kwargs=dict(k_size=7, **df)),
            CNNPrimitive(cls=ShuffleNetV2XceptionLayer, kwargs=dict(k_size=7, **df)),
            CNNPrimitive(cls=ShuffleNetV2XceptionLayer, kwargs=dict(k_size=7, **df)),
            CNNPrimitive(cls=ShuffleNetV2XceptionLayer, kwargs=dict(k_size=7, **df)),
            CNNPrimitive(cls=ShuffleNetV2XceptionLayer, kwargs=dict(k_size=7, **df)),
            CNNPrimitive(cls=ShuffleNetV2XceptionLayer, kwargs=dict(k_size=7, **df)),
            CNNPrimitive(cls=ShuffleNetV2XceptionLayer, kwargs=dict(k_size=7, **df)),
        ]
