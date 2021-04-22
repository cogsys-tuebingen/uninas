import torch
import torch.nn as nn
from uninas.modules.modules.cnn import GapSqueezeModule
from uninas.modules.layers.abstract import AbstractLayer, AbstractStepsLayer
from uninas.utils.misc import get_number
from uninas.utils.torch.misc import get_padding, get_splits
from uninas.utils.shape import Shape
from uninas.register import Register


class FactorizedReductionModule(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride=2):
        assert stride == 2
        super().__init__()
        s1, s2 = get_splits(c_out, 2, mode='even')
        self.conv_1 = nn.Conv2d(c_in, s1, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(c_in, s2, 1, stride=2, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)


@Register.network_layer()
class ClassificationLayer(AbstractStepsLayer):

    def __init__(self, dropout_rate=0.0, dropout_keep=True, bias=False, use_gap=True, **base_kwargs):
        base_kwargs['order'] = 'bn_w'
        super().__init__(dropout_rate=dropout_rate, dropout_keep=dropout_keep, **base_kwargs)
        self._add_to_kwargs(bias=bias, use_gap=use_gap)

    def _build(self, s_in: Shape, c_out: int, weight_functions=()) -> Shape:
        wf = list(weight_functions)
        if self.use_gap:
            wf += [GapSqueezeModule()]
        wf += [nn.Linear(s_in.num_features(), c_out, bias=self.bias)]
        return super()._build(s_in, c_out, weight_functions=wf)


@Register.network_layer()
class ZeroLayer(AbstractLayer):

    def __init__(self, stride=1, **base_kwargs):
        super().__init__(**base_kwargs)
        self._add_to_kwargs(stride=stride)

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        self._add_to_print_kwargs(features=c_out)
        return self.probe_outputs(s_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # replaces standard forward of BaseLayer, therefore no dropout/bn
        n, c, h, w = x.size()
        c = self.features
        h //= self.stride
        w //= self.stride
        with torch.no_grad():
            return torch.zeros(size=(n, c, h, w), device=x.device)


@Register.network_layer()
class PoolingLayer(AbstractStepsLayer):
    changes_c = False

    def __init__(self, pool_type='max', k_size=3, stride=1, padding='same', **base_kwargs):
        super().__init__(**base_kwargs)
        assert pool_type in ['max', 'avg']
        self._add_to_kwargs(pool_type=pool_type, k_size=k_size, stride=stride, padding=padding)

    def _build(self, s_in: Shape, c_out: int, weight_functions=()) -> Shape:
        padding = get_padding(self.padding, self.k_size, self.stride, 1)
        pool = (nn.AvgPool2d if self.pool_type == 'avg' else nn.MaxPool2d)(self.k_size, self.stride, padding)
        wf = list(weight_functions) + [pool]
        return super()._build(s_in, c_out, weight_functions=wf)


@Register.network_layer()
class PoolingConvLayer(AbstractStepsLayer):

    def __init__(self, pool_type='max', k_size=3, stride=1, padding='same', bias=False, **base_kwargs):
        super().__init__(**base_kwargs)
        assert pool_type in ['max', 'avg']
        self._add_to_kwargs(pool_type=pool_type, k_size=k_size, stride=stride, padding=padding, bias=bias)

    def _build(self, s_in: Shape, c_out: int, weight_functions=()) -> Shape:
        padding = get_padding(self.padding, self.k_size, self.stride, 1)
        pool = (nn.AvgPool2d if self.pool_type == 'avg' else nn.MaxPool2d)(self.k_size, self.stride, padding)
        conv = nn.Conv2d(s_in.num_features(), c_out, kernel_size=1, stride=1, padding=0, bias=self.bias)
        wf = list(weight_functions) + [pool, conv]
        return super()._build(s_in, c_out, weight_functions=wf)


@Register.network_layer()
class ConvLayer(AbstractStepsLayer):

    def __init__(self, k_size=3, dilation=1, stride=1, groups=1, bias=False, padding='same', **base_kwargs):
        super().__init__(**base_kwargs)
        self._add_to_kwargs(k_size=k_size, dilation=dilation, stride=stride, groups=groups, bias=bias, padding=padding)

    def _build(self, s_in: Shape, c_out: int, weight_functions=()) -> Shape:
        padding = get_padding(self.padding, self.k_size, self.stride, self.dilation)
        conv = nn.Conv2d(s_in.num_features(), c_out, kernel_size=self.k_size, stride=self.stride, padding=padding,
                         dilation=self.dilation, groups=get_number(self.groups, s_in.num_features()), bias=self.bias)
        wf = list(weight_functions) + [conv]
        return super()._build(s_in, c_out, weight_functions=wf)


@Register.network_layer()
class SepConvLayer(AbstractStepsLayer):

    def __init__(self, k_size=3, dilation=1, stride=1, groups=1, bias=False, padding='same', **base_kwargs):
        """
        Depthwise-separable convolution
        a spatial KxK kernel followed by a 1x1 kernel over all channels

        :param k_size: for the KxK layer
        :param dilation: for the KxK layer
        :param stride: for the KxK layer
        :param groups: for the 1x1 layer
        :param bias: for the 1x1 layer
        :param padding: for the KxK layer
        :param base_kwargs:
        """
        super().__init__(**base_kwargs)
        self._add_to_kwargs(k_size=k_size, dilation=dilation, stride=stride, groups=groups, bias=bias, padding=padding)

    def _build(self, s_in: Shape, c_out: int, weight_functions=()) -> Shape:
        padding = get_padding(self.padding, self.k_size, self.stride, self.dilation)
        depth_conv = nn.Conv2d(s_in.num_features(), s_in.num_features(), kernel_size=self.k_size, stride=self.stride,
                               padding=padding, dilation=self.dilation, groups=s_in.num_features(), bias=False)
        point_conv = nn.Conv2d(s_in.num_features(), c_out, kernel_size=1,
                               groups=get_number(self.groups, s_in.num_features()), bias=self.bias)
        wf = list(weight_functions) + [depth_conv, point_conv]
        return super()._build(s_in, c_out, weight_functions=wf)


@Register.network_layer()
class FactorizedReductionLayer(AbstractStepsLayer):

    def __init__(self, stride=2, **base_kwargs):
        super().__init__(**base_kwargs)
        assert stride == 2

    def _build(self, s_in: Shape, c_out: int, weight_functions=()) -> Shape:
        assert c_out % 2 == 0
        wf = list(weight_functions) + [FactorizedReductionModule(s_in.num_features(), c_out, stride=2)]
        return super()._build(s_in, c_out, weight_functions=wf)
