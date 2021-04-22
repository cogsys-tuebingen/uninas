"""
ShuffleNetV2+
https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2%2B
"""

import torch.nn as nn
from uninas.modules.networks.stackedcells import StackedCellsNetworkBody
from uninas.modules.stems.cnn import ConvStem
from uninas.modules.layers.cnn import ConvLayer
from uninas.modules.layers.shufflenet import ShuffleNetV2Layer, ShuffleNetV2XceptionLayer
from uninas.modules.heads.cnn import SeFeatureMixClassificationHead
from uninas.utils.shape import Shape
from uninas.utils.generate.networks.manually.abstract import get_stem_instance, get_head_instance,\
    get_passthrough_partials, get_network


def get_shufflenet_v2plus_medium(s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    stem = get_stem_instance(ConvStem, k_size=3, features=16, act_fun='hswish', stride=2, use_bn=True, bn_affine=True,
                            order='w_bn_act')
    head = get_head_instance(SeFeatureMixClassificationHead, se_cmul=0.25, se_act_fun='relu', se_squeeze_bias=True,
                            se_bn=True, se_excite_bias=False,
                            features=1280, act_fun='hswish', bias0=False, dropout=0.0, bias1=False)

    defaults = dict(padding='same', dilation=1, bn_affine=True, act_inplace=False, expansion=1)
    att = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=False,
               c_mul=0.25, squeeze_act='relu', excite_act='relu6', divisible=8,
               squeeze_bias=False, excite_bias=False, squeeze_bn=True, squeeze_bn_affine=True)

    cell_partials, cell_order = get_passthrough_partials([
        (48, ShuffleNetV2Layer,          defaults, dict(stride=2, k_size=3, act_fun='relu')),
        (48, ShuffleNetV2Layer,          defaults, dict(stride=1, k_size=3, act_fun='relu')),
        (48, ShuffleNetV2XceptionLayer,  defaults, dict(stride=1, k_size=3, act_fun='relu')),
        (48, ShuffleNetV2Layer,          defaults, dict(stride=1, k_size=5, act_fun='relu')),

        (128, ShuffleNetV2Layer,         defaults, dict(stride=2, k_size=5, act_fun='hswish')),
        (128, ShuffleNetV2Layer,         defaults, dict(stride=1, k_size=5, act_fun='hswish')),
        (128, ShuffleNetV2Layer,         defaults, dict(stride=1, k_size=3, act_fun='hswish')),
        (128, ShuffleNetV2Layer,         defaults, dict(stride=1, k_size=3, act_fun='hswish')),

        (256, ShuffleNetV2Layer,         defaults, dict(stride=2, k_size=7, act_fun='hswish', att_dict=att)),
        (256, ShuffleNetV2Layer,         defaults, dict(stride=1, k_size=3, act_fun='hswish', att_dict=att)),
        (256, ShuffleNetV2Layer,         defaults, dict(stride=1, k_size=7, act_fun='hswish', att_dict=att)),
        (256, ShuffleNetV2Layer,         defaults, dict(stride=1, k_size=5, act_fun='hswish', att_dict=att)),
        (256, ShuffleNetV2Layer,         defaults, dict(stride=1, k_size=5, act_fun='hswish', att_dict=att)),
        (256, ShuffleNetV2Layer,         defaults, dict(stride=1, k_size=3, act_fun='hswish', att_dict=att)),
        (256, ShuffleNetV2Layer,         defaults, dict(stride=1, k_size=7, act_fun='hswish', att_dict=att)),
        (256, ShuffleNetV2Layer,         defaults, dict(stride=1, k_size=3, act_fun='hswish', att_dict=att)),

        (512, ShuffleNetV2Layer,         defaults, dict(stride=2, k_size=7, act_fun='hswish', att_dict=att)),
        (512, ShuffleNetV2Layer,         defaults, dict(stride=1, k_size=5, act_fun='hswish', att_dict=att)),
        (512, ShuffleNetV2XceptionLayer, defaults, dict(stride=1, k_size=3, act_fun='hswish', att_dict=att)),
        (512, ShuffleNetV2Layer,         defaults, dict(stride=1, k_size=7, act_fun='hswish', att_dict=att)),

        (1280, ConvLayer, dict(), dict(k_size=1, bias=False, act_fun='hswish', act_inplace=True, order='w_bn_act',
                                       use_bn=True, bn_affine=True)),
    ])

    return get_network(StackedCellsNetworkBody, stem, head, cell_partials, cell_order, s_in, s_out)


if __name__ == '__main__':
    from uninas.utils.torch.misc import count_parameters
    from uninas.builder import Builder

    Builder()
    net = get_shufflenet_v2plus_medium().cuda()
    net.eval()
    print(net)
    print(count_parameters(net), count_parameters(net) - count_parameters(net.cells[:-1]))
