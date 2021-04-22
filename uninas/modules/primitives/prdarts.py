from uninas.modules.primitives.abstract import CNNPrimitive, DifferentConfigPrimitive, StrideChoiceCNNPrimitive, PrimitiveSet
from uninas.modules.layers.common import SkipLayer
from uninas.modules.layers.cnn import SepConvLayer, PoolingLayer, FactorizedReductionLayer
from uninas.register import Register


@Register.primitive_set()
class DNU_PRDartsPrimitives(PrimitiveSet):
    """
    Prune and Replace NAS
    https://arxiv.org/abs/1906.07528

    this is only a subset of all explorable functions, enough to build the PR_DARTS_DL1 and PR_DARTS_DL2 models
    since they would all be instanced at the same time, this implementation of the search space is not efficient to
    search through, it's just to generate the final models
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
            CNNPrimitive(cls=SepConvLayer, kwargs=dict(k_size=3, dilation=1, **act, **df)),
            CNNPrimitive(cls=SepConvLayer, kwargs=dict(k_size=5, dilation=1, **act, **df)),
            CNNPrimitive(cls=SepConvLayer, kwargs=dict(k_size=7, dilation=1, **act, **df)),
            CNNPrimitive(cls=SepConvLayer, kwargs=dict(k_size=7, dilation=1, **act, **df), stacked=2),
        ]
