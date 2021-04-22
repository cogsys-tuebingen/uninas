from uninas.modules.primitives.abstract import CNNPrimitive, PrimitiveSet
from uninas.modules.layers.cnn import ConvLayer, PoolingLayer
from uninas.register import Register


@Register.primitive_set()
class Bench101Primitives(PrimitiveSet):
    """
    NASBench: A Neural Architecture Search Dataset and Benchmark
    https://arxiv.org/abs1902.09635
    https://github.com/google-research/nasbench

    Conv3x3, Conv1x1, MaxPool3x3
    """

    @classmethod
    def get_primitives(cls, **primitive_kwargs) -> [CNNPrimitive]:
        act = dict(act_fun='relu', order='w_bn_act', act_inplace=False, bn_affine=False, use_bn=True)
        return [
            CNNPrimitive(cls=ConvLayer, kwargs=dict(k_size=3, dilation=1, **act)),
            CNNPrimitive(cls=ConvLayer, kwargs=dict(k_size=1, dilation=1, **act)),
            CNNPrimitive(PoolingLayer, kwargs=dict(k_size=3, pool_type='max', act_fun=None, order='w', use_bn=False)),
        ]

    @classmethod
    def get_order(cls) -> [str]:
        """ order of operations, using the bench101 names """
        return ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
