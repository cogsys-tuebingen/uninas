"""
using primitives/layers/modules from various papers
"""

from uninas.model.primitives.abstract import CNNPrimitive, StrideChoiceCNNPrimitive, PrimitiveSet
from uninas.model.layers.cnn import FactorizedReductionLayer
from uninas.model.layers.mobilenet import MobileInvertedConvLayer
from uninas.model.layers.scarletnas import LinearTransformerLayer
from uninas.register import Register


@Register.primitive_set()
class UninasMixedEcaHswishPrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks with Swish and EfficientChannelAttention everywhere
    expansion size {3, 6}
    mixed kernel size {(3), (3, 5), (3, 5, 7)}    (see MixNet)
    can also skip the block using a linear transformer / factorized reduction layer
    """

    @classmethod
    def _primitives(cls) -> [CNNPrimitive]:
        df = dict(dilation=1, act_inplace=True, bn_affine=True, act_fun='hswish')
        att_dict = dict(att_cls='EfficientChannelAttentionModule', use_c_substitute=False,
                        k_size=-1, gamma=2, b=1, excite_act='sigmoid')

        return [
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(expansion=3, att_dict=att_dict, k_size=(3,), **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(expansion=3, att_dict=att_dict, k_size=(3, 5), **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(expansion=3, att_dict=att_dict, k_size=(3, 5, 7), **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(expansion=6, att_dict=att_dict, k_size=(3,), **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(expansion=6, att_dict=att_dict, k_size=(3, 5), **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(expansion=6, att_dict=att_dict, k_size=(3, 5, 7), **df)),

            StrideChoiceCNNPrimitive([
                CNNPrimitive(cls=LinearTransformerLayer, kwargs=dict()),
                CNNPrimitive(cls=FactorizedReductionLayer, kwargs=dict(use_bn=False, order='w', act_fun='identity'))
            ]),
        ]
