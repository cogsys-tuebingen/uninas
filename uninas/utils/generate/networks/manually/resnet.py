"""
Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385
"""

from typing import Type
import torch.nn as nn
from uninas.modules.networks.stackedcells import StackedCellsNetworkBody
from uninas.modules.stems.cnn import ConvStem
from uninas.modules.layers.cnn import PoolingLayer
from uninas.modules.layers.resnet import AbstractResNetLayer, ResNetLayer, ResNetBottleneckLayer
from uninas.modules.heads.cnn import ClassificationHead
from uninas.utils.shape import Shape
from uninas.utils.generate.networks.manually.abstract import get_stem_instance, get_head_instance,\
    get_passthrough_partials, get_network


def _resnet(block: Type[AbstractResNetLayer], stages=(2, 2, 2, 2), inner_channels=(64, 128, 256, 512), expansion=1,
            s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    stem = get_stem_instance(ConvStem, features=inner_channels[0], stride=2, k_size=7, act_fun='relu')
    head = get_head_instance(ClassificationHead, bias=True, dropout=0.0)
    layers = [(inner_channels[0], PoolingLayer,
               dict(pool_type='max', k_size=3, padding='same', order='w', dropout_rate=0), dict(stride=2))]

    channels = [int(c*expansion) for c in inner_channels]
    defaults = dict(k_size=3, stride=1, padding='same', dilation=1, bn_affine=True, act_fun='relu', act_inplace=True,
                    expansion=1/expansion, has_first_act=False)
    for s, (num, cx) in enumerate(zip(stages, channels)):
        for i in range(num):
            if s > 0 and i == 0:
                layers.append((cx, block, defaults, dict(stride=2, shortcut_type='conv1x1')))
            elif i == 0 and expansion > 1:
                layers.append((cx, block, defaults, dict(stride=1, shortcut_type='conv1x1')))
            else:
                layers.append((cx, block, defaults, dict(stride=1, shortcut_type='id')))

    cell_partials, cell_order = get_passthrough_partials(layers)
    return get_network(StackedCellsNetworkBody, stem, head, cell_partials, cell_order, s_in, s_out)


def get_resnet18(s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    return _resnet(block=ResNetLayer, stages=(2, 2, 2, 2), expansion=1, s_in=s_in, s_out=s_out)


def get_resnet34(s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    return _resnet(block=ResNetLayer, stages=(3, 4, 6, 3), expansion=1, s_in=s_in, s_out=s_out)


def get_resnet50(s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    return _resnet(block=ResNetBottleneckLayer, stages=(3, 4, 6, 3), expansion=4, s_in=s_in, s_out=s_out)


def get_resnet101(s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    return _resnet(block=ResNetBottleneckLayer, stages=(3, 4, 23, 3), expansion=4, s_in=s_in, s_out=s_out)


def get_resnet152(s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    return _resnet(block=ResNetBottleneckLayer, stages=(3, 8, 36, 3), expansion=4, s_in=s_in, s_out=s_out)


if __name__ == '__main__':
    from uninas.utils.torch.misc import count_parameters
    from uninas.builder import Builder

    Builder()
    net = get_resnet50().cuda()
    net.eval()
    print(net)
    print('params', count_parameters(net))
    print('cell params', count_parameters(net.cells))

    for j, cell in enumerate(net.cells):
        print(j, count_parameters(cell))
