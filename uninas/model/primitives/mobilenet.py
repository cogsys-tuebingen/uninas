"""
MobileNetV2: Inverted Residuals and Linear Bottlenecks
https://arxiv.org/abs/1801.04381

also used e.g. in ProxylessNAS or FairNAS
"""

from uninas.model.primitives.abstract import PrimitiveSet
from uninas.model.layers.mobilenet import MixedMobileInvertedConvLayer
from uninas.register import Register


@Register.primitive_set()
class MobileNetV2Primitives(PrimitiveSet):
    """
    MobileNetV2 blocks
    expansion size {3, 6}
    kernel size {3, 5, 7}
    """

    @classmethod
    def mixed_instance(cls, name: str, strategy_name='default', **primitive_kwargs) -> MixedMobileInvertedConvLayer:
        return MixedMobileInvertedConvLayer(name=name, strategy_name=strategy_name, skip_op=None,
                                            k_sizes=(3, 5, 7), expansions=(3, 6),
                                            padding='same', dilation=1, bn_affine=True,
                                            act_fun='relu6', act_inplace=True, att_dict=None,
                                            **primitive_kwargs)


@Register.primitive_set()
class MobileNetV2SkipPrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks
    expansion size {3, 6}
    kernel size {3, 5, 7}
    can skip the block
    """

    @classmethod
    def mixed_instance(cls, name: str, strategy_name='default', **primitive_kwargs) -> MixedMobileInvertedConvLayer:
        return MixedMobileInvertedConvLayer(name=name, strategy_name=strategy_name, skip_op='SkipLayer',
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
    def mixed_instance(cls, name: str, strategy_name='default', **primitive_kwargs) -> MixedMobileInvertedConvLayer:
        return MixedMobileInvertedConvLayer(name=name, strategy_name=strategy_name, skip_op='LinearTransformerLayer',
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

    @classmethod
    def mixed_instance(cls, name: str, strategy_name='default', **primitive_kwargs) -> MixedMobileInvertedConvLayer:
        att_dict = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=True,
                        c_mul=0.25, squeeze_act='swish', excite_act='sigmoid',
                        squeeze_bias=True, excite_bias=True, squeeze_bn=False)
        return MixedMobileInvertedConvLayer(name=name, strategy_name=strategy_name, skip_op=None,
                                            k_sizes=(3, 5, 7), expansions=(3, 6),
                                            padding='same', dilation=1, bn_affine=True,
                                            act_fun='swish', act_inplace=True,
                                            att_dict=att_dict,
                                            **primitive_kwargs)
