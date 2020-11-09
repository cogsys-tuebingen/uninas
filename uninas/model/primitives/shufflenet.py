"""
Single Path One-Shot Neural Architecture Search with Uniform Sampling
https://arxiv.org/abs/1904.00420
"""

from uninas.model.primitives.abstract import CNNPrimitive, PrimitiveSet
from uninas.model.layers.shufflenet import ShuffleNetV2Layer, ShuffleNetV2XceptionLayer
from uninas.register import Register


@Register.primitive_set()
class ShuffleNetV2Primitives(PrimitiveSet):
    """
    ShuffleNetV2 blocks
    kernel size {3, 5, 7} or a ShuffleNetV2Xception block with kernel size 3
    """

    @classmethod
    def _primitives(cls) -> [CNNPrimitive]:
        df = dict(dilation=1, bn_affine=True, act_fun='relu', act_inplace=False, expansion=1.0, att_dict=None)
        return [
            CNNPrimitive(cls=ShuffleNetV2Layer, kwargs=dict(k_size=3, **df)),
            CNNPrimitive(cls=ShuffleNetV2Layer, kwargs=dict(k_size=5, **df)),
            CNNPrimitive(cls=ShuffleNetV2Layer, kwargs=dict(k_size=7, **df)),
            CNNPrimitive(cls=ShuffleNetV2XceptionLayer, kwargs=dict(k_size=3, **df)),
        ]
