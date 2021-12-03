"""
NAS-Bench-Macro
https://github.com/xiusu/NAS-Bench-Macro
"""

from uninas.modules.primitives.abstract import PrimitiveSet, CNNPrimitive, StrideChoiceCNNPrimitive
from uninas.modules.layers.common import SkipLayer
from uninas.modules.layers.cnn import ConvLayer
# from uninas.modules.layers.cnn import ZeroLayer
from uninas.modules.layers.mobilenet import MobileInvertedConvLayer
from uninas.register import Register


@Register.primitive_set()
class BenchMacroPrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks, skip connection
    ['id', 'ir_3x3_t3', 'ir_5x5_t6']
    """

    @classmethod
    def get_primitives(cls, stride=1, **primitive_kwargs) -> [CNNPrimitive]:
        df = dict(dilation=1, act_inplace=True, bn_affine=True, act_fun='relu', att_dict=None, stride=stride, fused=False)
        return [
            StrideChoiceCNNPrimitive([
                CNNPrimitive(cls=SkipLayer, kwargs=dict()),
                CNNPrimitive(cls=ConvLayer, kwargs=dict(k_size=1, bn_affine=True))
            ]),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=5, expansion=6.0, **df)),
            # CNNPrimitive(cls=ZeroLayer, kwargs=dict()),
        ]
