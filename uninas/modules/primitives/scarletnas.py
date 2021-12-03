"""
SCARLET-NAS: Bridging the gap between Stability and Scalability in Weight-sharing Neural Architecture Search
https://arxiv.org/abs/1908.06022

The paper is unclear whether the 13 primitives exist in parallel,
or if the non-SE and its respective SE version share weights.
"""

from uninas.modules.primitives.abstract import CNNPrimitive, DifferentConfigPrimitive, StrideChoiceCNNPrimitive, PrimitiveSet
from uninas.modules.layers.cnn import SepConvLayer, PoolingLayer, FactorizedReductionLayer
from uninas.modules.layers.mobilenet import MobileInvertedConvLayer
from uninas.modules.layers.scarletnas import LinearTransformerLayer
from uninas.register import Register


# as described in the DARTS paper, no zeros, replacing skip with linear transformers during the search (ScarletNAS)
@Register.primitive_set()
class ScarletDartsPrimitives(PrimitiveSet):
    """
    SepConv3x3, SepConv5x5, dilated SepConv3x3, dilated SepConv5x5, MaxPool3x3, AvgPool3x3, Skip/FactorizedReduction, Zero,
    using linear transformers for skip
    """

    @classmethod
    def get_primitives(cls, **primitive_kwargs) -> [CNNPrimitive]:
        act = dict(act_fun='relu', order='act_w_bn')
        df = dict(act_inplace=False, bn_affine=True, use_bn=True)
        dfnb = df.copy()
        dfnb['use_bn'] = False
        return [
            CNNPrimitive(cls=SepConvLayer, kwargs=dict(k_size=3, dilation=1, **act, **df), stacked=2),
            CNNPrimitive(cls=SepConvLayer, kwargs=dict(k_size=5, dilation=1, **act, **df), stacked=2),
            CNNPrimitive(cls=SepConvLayer, kwargs=dict(k_size=3, dilation=2, **act, **df)),
            CNNPrimitive(cls=SepConvLayer, kwargs=dict(k_size=5, dilation=2, **act, **df)),
            DifferentConfigPrimitive(
                CNNPrimitive(PoolingLayer, kwargs=dict(k_size=3, pool_type='max', act_fun=None, order='w_bn', **df)),
                CNNPrimitive(PoolingLayer, kwargs=dict(k_size=3, pool_type='max', act_fun=None, order='w', **dfnb))),
            DifferentConfigPrimitive(
                CNNPrimitive(PoolingLayer, kwargs=dict(k_size=3, pool_type='avg', act_fun=None, order='w_bn', **df)),
                CNNPrimitive(PoolingLayer, kwargs=dict(k_size=3, pool_type='avg', act_fun=None, order='w', **dfnb))),
            StrideChoiceCNNPrimitive([
                CNNPrimitive(cls=LinearTransformerLayer, kwargs=dict()),
                CNNPrimitive(cls=FactorizedReductionLayer, kwargs=dict(**act, **df))
            ]),
        ]


@Register.primitive_set()
class ScarletPrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks
    expansion size {3, 6}
    kernel size {3, 5, 7}
    Squeeze+Excitation {used, not used}
    can also skip the block using a linear transformer
    """

    @classmethod
    def get_primitives(cls, stride=1, **primitive_kwargs) -> [CNNPrimitive]:
        df = dict(dilation=1, act_inplace=True, bn_affine=True, act_fun='hswish', att_dict=None, fused=False)
        df_att = df.copy()
        df_att['att_dict'] = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=False,
                                  c_mul=0.25, squeeze_act='relu', excite_act='hsigmoid',
                                  squeeze_bias=True, excite_bias=True, squeeze_bn=False)
        primitives = [
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=5, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=7, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=6.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=5, expansion=6.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=7, expansion=6.0, **df)),

            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=3.0, **df_att)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=5, expansion=3.0, **df_att)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=7, expansion=3.0, **df_att)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=6.0, **df_att)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=5, expansion=6.0, **df_att)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=7, expansion=6.0, **df_att)),
        ]
        if stride == 1:
            primitives.append(CNNPrimitive(cls=LinearTransformerLayer, kwargs=dict()))
        return primitives
