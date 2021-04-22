"""
DARTS: Differentiable Architecture Search
https://arxiv.org/abs/1806.09055
"""

from uninas.modules.primitives.abstract import CNNPrimitive, DifferentConfigPrimitive, StrideChoiceCNNPrimitive, PrimitiveSet
from uninas.modules.layers.common import SkipLayer
from uninas.modules.layers.cnn import ZeroLayer, SepConvLayer, PoolingLayer, FactorizedReductionLayer
from uninas.register import Register


# as described in the DARTS paper, remove BN of pooling layers when finalizing
@Register.primitive_set()
class DartsPrimitives(PrimitiveSet):
    """
    SepConv3x3, SepConv5x5, dilated SepConv3x3, dilated SepConv5x5, MaxPool3x3, AvgPool3x3, Skip/FactorizedReduction, Zero
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
                CNNPrimitive(cls=SkipLayer, kwargs=dict()),
                CNNPrimitive(cls=FactorizedReductionLayer, kwargs=dict(**act, **df))
            ]),
            CNNPrimitive(cls=ZeroLayer, kwargs=dict()),
        ]


# as described in the DARTS paper, not using zeros as DARTS ignores them anyways
@Register.primitive_set()
class DartsNzPrimitives(PrimitiveSet):
    """
    SepConv3x3, SepConv5x5, dilated SepConv3x3, dilated SepConv5x5, MaxPool3x3, AvgPool3x3, Skip/FactorizedReduction
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
                CNNPrimitive(cls=SkipLayer, kwargs=dict()),
                CNNPrimitive(cls=FactorizedReductionLayer, kwargs=dict(**act, **df))
            ]),
        ]


@Register.primitive_set()
class DartsBnPrimitives(PrimitiveSet):
    """
    SepConv3x3, SepConv5x5, dilated SepConv3x3, dilated SepConv5x5, MaxPool3x3, AvgPool3x3, Skip/FactorizedReduction, Zero
    keep the batchnorm of pooling layers after the search
    """

    @classmethod
    def get_primitives(cls, **primitive_kwargs) -> [CNNPrimitive]:
        act = dict(act_fun='relu', order='act_w_bn')
        df = dict(act_inplace=False, bn_affine=True, use_bn=True)
        return [
            CNNPrimitive(cls=SepConvLayer, kwargs=dict(k_size=3, dilation=1, **act, **df), stacked=2),
            CNNPrimitive(cls=SepConvLayer, kwargs=dict(k_size=5, dilation=1, **act, **df), stacked=2),
            CNNPrimitive(cls=SepConvLayer, kwargs=dict(k_size=3, dilation=2, **act, **df)),
            CNNPrimitive(cls=SepConvLayer, kwargs=dict(k_size=5, dilation=2, **act, **df)),
            CNNPrimitive(PoolingLayer, kwargs=dict(k_size=3, pool_type='max', act_fun=None, order='w_bn', **df)),
            CNNPrimitive(PoolingLayer, kwargs=dict(k_size=3, pool_type='avg', act_fun=None, order='w_bn', **df)),
            StrideChoiceCNNPrimitive([
                CNNPrimitive(cls=SkipLayer, kwargs=dict()),
                CNNPrimitive(cls=FactorizedReductionLayer, kwargs=dict(**act, **df))
            ]),
            CNNPrimitive(cls=ZeroLayer, kwargs=dict()),
        ]
