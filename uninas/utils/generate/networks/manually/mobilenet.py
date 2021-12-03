"""
MobileNetV2: Inverted Residuals and Linear Bottlenecks
https://arxiv.org/abs/1801.04381

Searching for MobileNetV3
https://arxiv.org/abs/1905.02244
"""

import torch.nn as nn
from uninas.modules.networks.stackedcells import StackedCellsNetworkBody
from uninas.modules.stems.mobilenet import MobileNetV2Stem
from uninas.modules.layers.cnn import ConvLayer
from uninas.modules.layers.mobilenet import MobileInvertedConvLayer
from uninas.modules.heads.cnn import FeatureMixClassificationHead
from uninas.utils.shape import Shape
from uninas.utils.generate.networks.manually.abstract import get_stem_instance, get_head_instance,\
    get_passthrough_partials, get_network


def get_mobilenet_v2(s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    stem = get_stem_instance(MobileNetV2Stem, features=32, features1=16, act_fun='relu6', act_fun1='relu6')
    head = get_head_instance(FeatureMixClassificationHead, features=1280, act_fun='relu6')

    defaults = dict(k_size=3, stride=1, padding='same', expansion=6, dilation=1, bn_affine=True,
                    act_fun='relu6', act_inplace=True, att_dict=None, fused=False)
    cell_partials, cell_order = get_passthrough_partials([
        (24, MobileInvertedConvLayer, defaults, dict(stride=2)),
        (24, MobileInvertedConvLayer, defaults, dict(stride=1)),

        (32, MobileInvertedConvLayer, defaults, dict(stride=2)),
        (32, MobileInvertedConvLayer, defaults, dict(stride=1)),
        (32, MobileInvertedConvLayer, defaults, dict(stride=1)),

        (64, MobileInvertedConvLayer, defaults, dict(stride=2)),
        (64, MobileInvertedConvLayer, defaults, dict(stride=1)),
        (64, MobileInvertedConvLayer, defaults, dict(stride=1)),
        (64, MobileInvertedConvLayer, defaults, dict(stride=1)),

        (96, MobileInvertedConvLayer, defaults, dict(stride=1)),
        (96, MobileInvertedConvLayer, defaults, dict(stride=1)),
        (96, MobileInvertedConvLayer, defaults, dict(stride=1)),

        (160, MobileInvertedConvLayer, defaults, dict(stride=2)),
        (160, MobileInvertedConvLayer, defaults, dict(stride=1)),
        (160, MobileInvertedConvLayer, defaults, dict(stride=1)),

        (320, MobileInvertedConvLayer, defaults, dict(stride=1)),
    ])

    return get_network(StackedCellsNetworkBody, stem, head, cell_partials, cell_order, s_in, s_out)


def get_mobilenet_v3_large100(s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    stem = get_stem_instance(MobileNetV2Stem, features=16, features1=16, act_fun='hswish', act_fun1='relu')
    head = get_head_instance(FeatureMixClassificationHead, features=1280, act_fun='hswish', gap_first=True, bias=True)

    # weird squeeze + excitation channel numbers
    defaults = dict(padding='same', dilation=1, bn_affine=True, act_inplace=True, fused=False)
    se0 = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=False,
               c_mul=0.33334, squeeze_act='relu', excite_act='sigmoid', divisible=8,
               squeeze_bias=True, excite_bias=True, squeeze_bn=False)
    se1 = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=False,
               c_mul=0.25, squeeze_act='relu', excite_act='sigmoid', divisible=8,
               squeeze_bias=True, excite_bias=True, squeeze_bn=False)

    cell_partials, cell_order = get_passthrough_partials([
        (24, MobileInvertedConvLayer, defaults, dict(stride=2, k_size=3, expansion=4, act_fun='relu')),
        (24, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=3, expansion=3, act_fun='relu')),

        (40, MobileInvertedConvLayer, defaults, dict(stride=2, k_size=5, expansion=3, act_fun='relu', att_dict=se0)),
        (40, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=5, expansion=3, act_fun='relu', att_dict=se1)),
        (40, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=5, expansion=3, act_fun='relu', att_dict=se1)),

        (80, MobileInvertedConvLayer, defaults, dict(stride=2, k_size=3, expansion=6, act_fun='hswish')),
        (80, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=3, expansion=2.5, act_fun='hswish')),
        (80, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=3, expansion=2.3, act_fun='hswish')),
        (80, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=3, expansion=2.3, act_fun='hswish')),

        (112, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=3, expansion=6, act_fun='hswish', att_dict=se1)),
        (112, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=3, expansion=6, act_fun='hswish', att_dict=se1)),

        (160, MobileInvertedConvLayer, defaults, dict(stride=2, k_size=5, expansion=6, act_fun='hswish', att_dict=se1)),
        (160, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=5, expansion=6, act_fun='hswish', att_dict=se1)),
        (160, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=5, expansion=6, act_fun='hswish', att_dict=se1)),

        (960, ConvLayer, dict(), dict(k_size=1, bias=False, act_fun='hswish', act_inplace=True, order='w_bn_act',
                                      use_bn=True, bn_affine=True)),
    ])

    return get_network(StackedCellsNetworkBody, stem, head, cell_partials, cell_order, s_in, s_out)


def get_mobilenet_v3_small100(s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    stem = get_stem_instance(MobileNetV2Stem, features=16, features1=16, act_fun='hswish', act_fun1='relu',
                             stride1=2, se_cmul1=0.5)
    head = get_head_instance(FeatureMixClassificationHead, features=1024, act_fun='hswish', gap_first=True, bias=True)

    defaults = dict(padding='same', dilation=1, bn_affine=True, act_inplace=True, fused=False)
    se = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=False,
              c_mul=0.25, squeeze_act='relu', excite_act='sigmoid', divisible=8,
              squeeze_bias=True, excite_bias=True, squeeze_bn=False)

    cell_partials, cell_order = get_passthrough_partials([
        (24, MobileInvertedConvLayer, defaults, dict(stride=2, k_size=3, expansion=4.5, act_fun='relu')),
        (24, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=3, expansion=3.5, act_fun='relu')),

        (40, MobileInvertedConvLayer, defaults, dict(stride=2, k_size=5, expansion=4, act_fun='hswish', att_dict=se)),
        (40, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=5, expansion=6, act_fun='hswish', att_dict=se)),
        (40, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=5, expansion=6, act_fun='hswish', att_dict=se)),

        (48, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=5, expansion=3, act_fun='hswish', att_dict=se)),
        (48, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=5, expansion=3, act_fun='hswish', att_dict=se)),

        (96, MobileInvertedConvLayer, defaults, dict(stride=2, k_size=5, expansion=6, act_fun='hswish', att_dict=se)),
        (96, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=5, expansion=6, act_fun='hswish', att_dict=se)),
        (96, MobileInvertedConvLayer, defaults, dict(stride=1, k_size=5, expansion=6, act_fun='hswish', att_dict=se)),

        (576, ConvLayer, dict(), dict(k_size=1, bias=False, act_fun='hswish', act_inplace=True, order='w_bn_act',
                                      use_bn=True, bn_affine=True)),
    ])

    return get_network(StackedCellsNetworkBody, stem, head, cell_partials, cell_order, s_in, s_out)
