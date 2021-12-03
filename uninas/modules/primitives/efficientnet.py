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

from uninas.modules.modules.abstract import AbstractModule
from uninas.modules.primitives.abstract import PrimitiveSet, CNNPrimitive
from uninas.modules.layers.mobilenet import MobileInvertedConvLayer
from uninas.modules.layers.scarletnas import LinearTransformerLayer
from uninas.modules.layers.mobilenet import SharedMixedMobileInvertedConvLayer
from uninas.register import Register


@Register.primitive_set()
class EfficientNetPrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks with Squeeze+Excitation and Swish
    expansion size {3, 6}
    kernel size {3, 5, 7}
    """

    @classmethod
    def get_primitives(cls, stride=1, **primitive_kwargs) -> [CNNPrimitive]:
        df = dict(dilation=1, act_inplace=True, bn_affine=True, act_fun='swish', fused=False)
        df['att_dict'] = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=True,
                              c_mul=0.25, squeeze_act='swish', excite_act='sigmoid',
                              squeeze_bias=True, excite_bias=True, squeeze_bn=False)
        primitives = [
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=5, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=7, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=6.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=5, expansion=6.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=7, expansion=6.0, **df)),
        ]
        if stride == 1:
            primitives.append(CNNPrimitive(cls=LinearTransformerLayer, kwargs=dict()))
        return primitives


@Register.primitive_set()
class SharedEfficientNetPrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks with Squeeze+Excitation and Swish
    use shared weights for the 1x1 convolutions when possible
    expansion size {3, 6}
    kernel size {3, 5, 7}
    """

    @classmethod
    def get_shared_instance_primitives(cls, name: str, strategy_name: str, **primitive_kwargs) -> [AbstractModule]:
        att_dict = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=True,
                        c_mul=0.25, squeeze_act='swish', excite_act='sigmoid',
                        squeeze_bias=True, excite_bias=True, squeeze_bn=False)
        shared = SharedMixedMobileInvertedConvLayer(name=name, strategy_name=strategy_name, skip_op=None,
                                                    k_sizes=(3, 5, 7), expansions=(3, 6),
                                                    padding='same', dilation=1, bn_affine=True,
                                                    act_fun='swish', act_inplace=True,
                                                    att_dict=att_dict,
                                                    **primitive_kwargs)
        return shared.get_paths_as_modules()


@Register.primitive_set()
class EfficientNetECAPrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks with Efficient Channel Attention and Swish
    expansion size {3, 6}
    kernel size {3, 5, 7}
    """

    @classmethod
    def get_primitives(cls, stride=1, **primitive_kwargs) -> [CNNPrimitive]:
        df = dict(dilation=1, act_inplace=True, bn_affine=True, act_fun='swish', fused=False)
        df['att_dict'] = dict(att_cls='EfficientChannelAttentionModule', use_c_substitute=True)
        primitives = [
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=5, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=7, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=6.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=5, expansion=6.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=7, expansion=6.0, **df)),
        ]
        if stride == 1:
            primitives.append(CNNPrimitive(cls=LinearTransformerLayer, kwargs=dict()))
        return primitives


@Register.primitive_set()
class SharedEfficientNetECAPrimitives(PrimitiveSet):
    """
    MobileNetV2 blocks with Efficient Channel Attention and Swish
    use shared weights for the 1x1 convolutions when possible
    expansion size {3, 6}
    kernel size {3, 5, 7}
    """

    @classmethod
    def get_shared_instance_primitives(cls, name: str, strategy_name: str, **primitive_kwargs) -> [AbstractModule]:
        att_dict = dict(att_cls='EfficientChannelAttentionModule', use_c_substitute=True)
        shared = SharedMixedMobileInvertedConvLayer(name=name, strategy_name=strategy_name, skip_op=None,
                                                    k_sizes=(3, 5, 7), expansions=(3, 6),
                                                    padding='same', dilation=1, bn_affine=True,
                                                    act_fun='swish', act_inplace=True,
                                                    att_dict=att_dict,
                                                    **primitive_kwargs)
        return shared.get_paths_as_modules()


@Register.primitive_set()
class EfficientNetPrimitivesMini(PrimitiveSet):
    """
    MobileNetV2 blocks with Squeeze+Excitation and Swish
    expansion size {3, 6}
    kernel size {3, 5}
    """

    @classmethod
    def get_primitives(cls, stride=1, **primitive_kwargs) -> [CNNPrimitive]:
        df = dict(dilation=1, act_inplace=True, bn_affine=True, act_fun='swish', fused=False)
        df['att_dict'] = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=True,
                              c_mul=0.25, squeeze_act='swish', excite_act='sigmoid',
                              squeeze_bias=True, excite_bias=True, squeeze_bn=False)
        return [
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=5, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=7, expansion=3.0, **df)),
            CNNPrimitive(cls=MobileInvertedConvLayer, kwargs=dict(k_size=3, expansion=6.0, **df)),
        ]
