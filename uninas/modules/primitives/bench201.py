from uninas.modules.primitives.abstract import CNNPrimitive, StrideChoiceCNNPrimitive, PrimitiveSet
from uninas.modules.layers.common import SkipLayer
from uninas.modules.layers.scarletnas import LinearTransformerLayer
from uninas.modules.layers.cnn import ZeroLayer, ConvLayer, PoolingLayer, PoolingConvLayer, FactorizedReductionLayer
from uninas.register import Register


@Register.primitive_set()
class Bench201Primitives(PrimitiveSet):
    """
    NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search
    https://arxiv.org/abs/2001.00326

    NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size
    https://arxiv.org/abs/2009.00437

    Zero, Skip, Conv1x1, Conv3x3, AvgPool3x3
    """

    @classmethod
    def get_primitives(cls, **primitive_kwargs) -> [CNNPrimitive]:
        act = dict(act_fun='relu', order='act_w_bn', act_inplace=False, bn_affine=False, use_bn=True)
        return [
            CNNPrimitive(cls=ZeroLayer, kwargs=dict()),
            StrideChoiceCNNPrimitive([
                CNNPrimitive(cls=SkipLayer, kwargs=dict()),
                CNNPrimitive(cls=FactorizedReductionLayer, kwargs=dict(**act))
            ]),
            CNNPrimitive(cls=ConvLayer, kwargs=dict(k_size=1, dilation=1, **act)),
            CNNPrimitive(cls=ConvLayer, kwargs=dict(k_size=3, dilation=1, **act)),
            CNNPrimitive(PoolingLayer, kwargs=dict(k_size=3, pool_type='avg', act_fun=None, order='w', use_bn=False)),
        ]

    @classmethod
    def get_order(cls) -> [str]:
        """ order of operations, using the bench201 names """
        return ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']


@Register.primitive_set()
class Bench201LTsPrimitives(Bench201Primitives):
    """
    Zero, Skip, Conv1x1, Conv3x3, AvgPool3x3
    using a linear transformer for Skip
    """

    @classmethod
    def get_primitives(cls, **primitive_kwargs) -> [CNNPrimitive]:
        act = dict(act_fun='relu', order='act_w_bn', act_inplace=False, bn_affine=False, use_bn=True)
        return [
            CNNPrimitive(cls=ZeroLayer, kwargs=dict()),
            StrideChoiceCNNPrimitive([
                CNNPrimitive(cls=LinearTransformerLayer, kwargs=dict()),
                CNNPrimitive(cls=FactorizedReductionLayer, kwargs=dict(**act))
            ]),
            CNNPrimitive(cls=ConvLayer, kwargs=dict(k_size=1, dilation=1, **act)),
            CNNPrimitive(cls=ConvLayer, kwargs=dict(k_size=3, dilation=1, **act)),
            CNNPrimitive(PoolingLayer, kwargs=dict(k_size=3, pool_type='avg', act_fun=None, order='w', use_bn=False)),
        ]


@Register.primitive_set()
class Bench201LTspPrimitives(Bench201Primitives):
    """
    Zero, Skip, Conv1x1, Conv3x3, AvgPool3x3
    using linear transformers for Skip and Pool
    """

    @classmethod
    def get_primitives(cls, **primitive_kwargs) -> [CNNPrimitive]:
        act = dict(act_fun='relu', order='act_w_bn', act_inplace=False, bn_affine=False, use_bn=True)
        return [
            CNNPrimitive(cls=ZeroLayer, kwargs=dict()),
            StrideChoiceCNNPrimitive([
                CNNPrimitive(cls=LinearTransformerLayer, kwargs=dict()),
                CNNPrimitive(cls=FactorizedReductionLayer, kwargs=dict(**act))
            ]),
            CNNPrimitive(cls=ConvLayer, kwargs=dict(k_size=1, dilation=1, **act)),
            CNNPrimitive(cls=ConvLayer, kwargs=dict(k_size=3, dilation=1, **act)),
            CNNPrimitive(PoolingConvLayer, kwargs=dict(k_size=3, pool_type='avg', act_fun=None,
                                                       order='w', bn_affine=False, use_bn=False, bias=False)),
        ]
