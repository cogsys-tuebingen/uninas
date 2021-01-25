"""
MobileNetV2: Inverted Residuals and Linear Bottlenecks
https://arxiv.org/abs/1801.04381

also used e.g. in ProxylessNAS or FairNAS
"""

from uninas.model.primitives.abstract import PrimitiveSet, CNNPrimitive
from uninas.model.layers.common import SkipLayer
from uninas.model.layers.mobilenet import MobileInvertedConvLayer, FusedMobileInvertedConvLayer
from uninas.model.layers.scarletnas import LinearTransformerLayer
from uninas.register import Register


@Register.primitive_set()
class MobileNetV2Primitives(PrimitiveSet):
    """
    MobileNetV2 blocks, fused
    expansion size {3, 6}
    kernel size {3, 5, 7}
    """

    @classmethod
    def get_primitives(cls, stride=1, **primitive_kwargs) -> [CNNPrimitive]:
        df = dict(dilation=1, act_inplace=True, bn_affine=True, act_fun='relu6', att_dict=None, stride=stride)
        return [
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=5, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=7, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=6.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=5, expansion=6.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=7, expansion=6.0, **df)),
        ]

    def _fused_instance(self, name: str, **primitive_kwargs) -> FusedMobileInvertedConvLayer:
        return FusedMobileInvertedConvLayer(name=name, strategy_name=self.strategy_name, skip_op=None,
                                            k_sizes=(3, 5, 7), expansions=(3, 6),
                                            padding='same', dilation=1, bn_affine=True,
                                            act_fun='relu6', act_inplace=True, att_dict=None,
                                            **primitive_kwargs)


@Register.primitive_set()
class MobileNetV2SkipPrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks, fused
    expansion size {3, 6}
    kernel size {3, 5, 7}
    can skip the block
    """

    @classmethod
    def get_primitives(cls, stride=1, **primitive_kwargs) -> [CNNPrimitive]:
        primitives = MobileNetV2Primitives.get_primitives(stride=stride, **primitive_kwargs)
        if stride == 1:
            primitives.append(CNNPrimitive(cls=SkipLayer, kwargs=dict())),
        return primitives

    def _fused_instance(self, name: str, **primitive_kwargs) -> FusedMobileInvertedConvLayer:
        return FusedMobileInvertedConvLayer(name=name, strategy_name=self.strategy_name, skip_op='SkipLayer',
                                            k_sizes=(3, 5, 7), expansions=(3, 6),
                                            padding='same', dilation=1, bn_affine=True,
                                            act_fun='relu6', act_inplace=True, att_dict=None,
                                            **primitive_kwargs)


@Register.primitive_set()
class MobileNetV2SkipLTPrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks
    expansion size {3, 6}
    kernel size {3, 5, 7}
    can skip the block via linear transformer
    """

    @classmethod
    def get_primitives(cls, stride=1, **primitive_kwargs) -> [CNNPrimitive]:
        primitives = MobileNetV2Primitives.get_primitives(stride=stride, **primitive_kwargs)
        if stride == 1:
            primitives.append(CNNPrimitive(cls=LinearTransformerLayer, kwargs=dict())),
        return primitives

    def _fused_instance(self, name: str, **primitive_kwargs) -> FusedMobileInvertedConvLayer:
        return FusedMobileInvertedConvLayer(name=name, strategy_name=self.strategy_name, skip_op='LinearTransformerLayer',
                                            k_sizes=(3, 5, 7), expansions=(3, 6),
                                            padding='same', dilation=1, bn_affine=True,
                                            act_fun='relu6', act_inplace=True, att_dict=None,
                                            **primitive_kwargs)


@Register.primitive_set()
class MobileNetV2SePrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks with Squeeze+Excitation
    expansion size {3, 6}
    kernel size {3, 5, 7}
    """

    def _fused_instance(self, name: str, **primitive_kwargs) -> FusedMobileInvertedConvLayer:
        att_dict = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=True,
                        c_mul=0.25, squeeze_act='swish', excite_act='sigmoid',
                        squeeze_bias=True, excite_bias=True, squeeze_bn=False)
        return FusedMobileInvertedConvLayer(name=name, strategy_name=self.strategy_name, skip_op=None,
                                            k_sizes=(3, 5, 7), expansions=(3, 6),
                                            padding='same', dilation=1, bn_affine=True,
                                            act_fun='swish', act_inplace=True,
                                            att_dict=att_dict,
                                            **primitive_kwargs)
