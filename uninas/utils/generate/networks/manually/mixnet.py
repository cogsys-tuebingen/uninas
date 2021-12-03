"""
MixConv: Mixed depthwise convolutional kernels
https://arxiv.org/abs/1907.09595

manually engineering as a search space according to the paper has
 - expansion groups {1, 2} for in/out 1x1 convs
 - expansion size (probably {3, 6}?)
 - squeeze-excitation (probably {None, 0.25, 0.5}?)
 - mixed kernel size {3, 3.5, 3.5.7, 3.5.7.9, 3.5.7.9.11}
 - activation function? relu/swish are used
which are at least 120 options per block in a naive implementation
(although repeated blocks suggest that they are topology-grouped within stages)
"""

import torch.nn as nn
from uninas.modules.networks.stackedcells import StackedCellsNetworkBody
from uninas.modules.stems.mobilenet import MobileNetV2Stem
from uninas.modules.layers.mobilenet import MobileInvertedConvLayer
from uninas.modules.heads.cnn import FeatureMixClassificationHead
from uninas.utils.shape import Shape
from uninas.utils.generate.networks.manually.abstract import get_stem_instance, get_head_instance,\
    get_passthrough_partials, get_network


def get_mixnet_s(s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    stem = get_stem_instance(MobileNetV2Stem, features=16, features1=16, act_fun='relu', act_fun1='relu')
    head = get_head_instance(FeatureMixClassificationHead, features=1536, act_fun='relu')

    defaults = dict(k_size=(3,), k_size_in=1, k_size_out=1, padding='same', dilation=1,
                    bn_affine=True, act_inplace=True, att_dict=None, fused=False)
    se25 = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=True,
                c_mul=0.25, squeeze_act='swish', excite_act='sigmoid',
                squeeze_bias=True, excite_bias=True, squeeze_bn=False)
    se5 = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=True,
               c_mul=0.5, squeeze_act='swish', excite_act='sigmoid',
               squeeze_bias=True, excite_bias=True, squeeze_bn=False)

    cell_partials, cell_order = get_passthrough_partials([
        (24, MobileInvertedConvLayer, defaults, dict(stride=2, expansion=6, act_fun='relu', k_size=(3,), k_size_in=(1, 1), k_size_out=(1, 1))),
        (24, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=3, act_fun='relu', k_size=(3,), k_size_in=(1, 1), k_size_out=(1, 1))),

        (40, MobileInvertedConvLayer, defaults, dict(stride=2, expansion=6, act_fun='swish', k_size=(3, 5, 7), att_dict=se5)),
        (40, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se5)),
        (40, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se5)),
        (40, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se5)),

        (80, MobileInvertedConvLayer, defaults, dict(stride=2, expansion=6, act_fun='swish', k_size=(3, 5, 7), k_size_out=(1, 1), att_dict=se25)),
        (80, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5), k_size_out=(1, 1), att_dict=se25)),
        (80, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5), k_size_out=(1, 1), att_dict=se25)),

        (120, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5, 7), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se5)),
        (120, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=3, act_fun='swish', k_size=(3, 5, 7, 9), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se5)),
        (120, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=3, act_fun='swish', k_size=(3, 5, 7, 9), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se5)),

        (200, MobileInvertedConvLayer, defaults, dict(stride=2, expansion=6, act_fun='swish', k_size=(3, 5, 7, 9, 11), att_dict=se5)),
        (200, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5, 7, 9), k_size_out=(1, 1), att_dict=se5)),
        (200, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5, 7, 9), k_size_out=(1, 1), att_dict=se5)),
    ])

    return get_network(StackedCellsNetworkBody, stem, head, cell_partials, cell_order, s_in, s_out)


def get_mixnet_m(s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    stem = get_stem_instance(MobileNetV2Stem, features=24, features1=24, act_fun='relu', act_fun1='relu')
    head = get_head_instance(FeatureMixClassificationHead, features=1536, act_fun='relu')

    defaults = dict(k_size=(3,), k_size_in=1, k_size_out=1, padding='same', dilation=1,
                    bn_affine=True, act_inplace=True, att_dict=None, fused=False)
    se25 = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=True,
                c_mul=0.25, squeeze_act='swish', excite_act='sigmoid',
                squeeze_bias=True, excite_bias=True, squeeze_bn=False)
    se5 = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=True,
               c_mul=0.5, squeeze_act='swish', excite_act='sigmoid',
               squeeze_bias=True, excite_bias=True, squeeze_bn=False)

    cell_partials, cell_order = get_passthrough_partials([
        (32, MobileInvertedConvLayer, defaults, dict(stride=2, expansion=6, act_fun='relu', k_size=(3, 5, 7), k_size_in=(1, 1), k_size_out=(1, 1))),
        (32, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=3, act_fun='relu', k_size=(3,), k_size_in=(1, 1), k_size_out=(1, 1))),

        (40, MobileInvertedConvLayer, defaults, dict(stride=2, expansion=6, act_fun='swish', k_size=(3, 5, 7, 9), att_dict=se5)),
        (40, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se5)),
        (40, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se5)),
        (40, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se5)),

        (80, MobileInvertedConvLayer, defaults, dict(stride=2, expansion=6, act_fun='swish', k_size=(3, 5, 7), att_dict=se25)),
        (80, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5, 7, 9), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se25)),
        (80, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5, 7, 9), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se25)),
        (80, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5, 7, 9), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se25)),

        (120, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3,), att_dict=se5)),
        (120, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=3, act_fun='swish', k_size=(3, 5, 7, 9), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se5)),
        (120, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=3, act_fun='swish', k_size=(3, 5, 7, 9), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se5)),
        (120, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=3, act_fun='swish', k_size=(3, 5, 7, 9), k_size_in=(1, 1), k_size_out=(1, 1), att_dict=se5)),

        (200, MobileInvertedConvLayer, defaults, dict(stride=2, expansion=6, act_fun='swish', k_size=(3, 5, 7, 9), att_dict=se5)),
        (200, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5, 7, 9), k_size_out=(1, 1), att_dict=se5)),
        (200, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5, 7, 9), k_size_out=(1, 1), att_dict=se5)),
        (200, MobileInvertedConvLayer, defaults, dict(stride=1, expansion=6, act_fun='swish', k_size=(3, 5, 7, 9), k_size_out=(1, 1), att_dict=se5)),
    ])

    return get_network(StackedCellsNetworkBody, stem, head, cell_partials, cell_order, s_in, s_out)
