"""
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
https://arxiv.org/abs/1905.11946

note that this is changed to be more in line with the other search spaces,
the actual search space is based on MnasNet which also contains:
    - regular conv
    - depthwise separable conv
    - optional skip ops
but only kernel sizes of sizes 3x3 and 5x5 and no SE/swish
"""

from uninas.model.primitives.abstract import PrimitiveSet
from uninas.model.layers.mobilenet import MixedMobileInvertedConvLayer
from uninas.register import Register


@Register.primitive_set()
class EfficientNetPrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks with Squeeze+Excitation and Swish
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


@Register.primitive_set()
class EfficientNetECAPrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks with Efficient Channel Attention and Swish
    expansion size {3, 6}
    kernel size {3, 5, 7}
    """

    @classmethod
    def mixed_instance(cls, name: str, strategy_name='default', **primitive_kwargs) -> MixedMobileInvertedConvLayer:
        att_dict = dict(att_cls='EfficientChannelAttentionModule', use_c_substitute=True)
        return MixedMobileInvertedConvLayer(name=name, strategy_name=strategy_name, skip_op=None,
                                            k_sizes=(3, 5, 7), expansions=(3, 6),
                                            padding='same', dilation=1, bn_affine=True,
                                            act_fun='swish', act_inplace=True,
                                            att_dict=att_dict,
                                            **primitive_kwargs)
