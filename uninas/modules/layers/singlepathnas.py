"""
Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours
https://arxiv.org/abs/1904.02877
"""

import torch
import torch.nn as nn
from uninas.modules.layers.abstract import AbstractLayer, AbstractStepsLayer
from uninas.modules.layers.cnn import ConvLayer, SepConvLayer
from uninas.modules.layers.mobilenet import MobileInvertedConvLayer
from uninas.modules.modules.misc import DropPathModule
from uninas.utils.torch.misc import get_padding, make_divisible
from uninas.utils.shape import Shape
from uninas.utils.misc import get_number
from uninas.register import Register


class TrainableMask(nn.Module):
    def __init__(self, masks: [torch.Tensor]):
        """
        A trainable mask that learns how many of its masks should be applied
        :param masks: list of disjoint masks
        """
        super().__init__()
        self.has_mask = len(masks) > 1
        self.register_buffer('masks', torch.stack(masks, dim=0) if self.has_mask else None)
        self.thresholds = nn.Parameter(torch.zeros(len(masks)), requires_grad=True) if self.has_mask else None

    def _mask(self, weight: torch.Tensor, masks: [torch.Tensor], thresholds: [torch.Tensor]):
        if not masks or not thresholds:
            return 0
        m = masks.pop(0)
        t = thresholds.pop(0)
        norm = torch.norm(weight * m)
        indicator = ((norm > t).float() - torch.sigmoid(norm - t)).detach() + torch.sigmoid(norm - t)
        return indicator * (m + self._mask(weight, masks, thresholds))

    def forward(self, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.has_mask:
            mask = mask * self._mask(weight, list(self.masks), list(self.thresholds))
        return mask

    def get_finalized(self, weight: torch.Tensor) -> int:
        if self.has_mask:
            for i, (m, t) in enumerate(zip(self.masks, self.thresholds)):
                if torch.norm(weight * m) < t:
                    return i
            return len(self.masks)-1
        return 0


class SuperKernelThresholdConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, k_sizes=(3, 5, 7), c_multipliers=(0.5, 1.0),
                 dilation=1, stride=1, padding='same', groups=1, bias=False):
        """
        A super-kernel that applies convolution with a masked weight, using differentiable weights and thresholds
        to figure out the best masking, thus kernel size and num output channels.
        Since the mask is learned, rather than generating different outputs, this module can be used efficiently to
        learn the architecture of (huge) networks.

        :param c_in: num input channels
        :param c_out: num output channels
        :param k_sizes: kernel sizes
        :param c_multipliers:
        :param dilation: dilation for the kernel
        :param stride: stride for the kernel
        :param padding: 'same' or number
        :param bias: whether to use a bias
        """
        super().__init__()
        k_sizes = sorted(k_sizes)
        max_k = max(k_sizes)
        c_multipliers = sorted(c_multipliers)
        assert max(c_multipliers) == 1.0, "Can only reduce max channels, choose a higher c_in/c_out"

        self.c_in = c_in
        self.c_out = c_out
        self.k_sizes = k_sizes
        self.c_multipliers = c_multipliers
        self.c_out_list = [int(cm * c_out) for cm in c_multipliers]
        self._padding = get_padding(padding, max_k, stride, 1)
        self._stride = stride
        self._dilation = dilation
        self._groups = get_number(groups, c_out)
        assert c_in % self._groups == 0

        # conv and bias weights
        self.weight = nn.Parameter(torch.zeros(c_out, c_in // self._groups, max_k, max_k), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(c_out), requires_grad=True) if bias else None
        nn.init.kaiming_normal_(self.weight, mode='fan_out')

        # channel masks
        masks_c = []
        for cs in self.c_out_list:
            mask = torch.ones(size=(c_out, 1, 1, 1), dtype=self.weight.dtype)
            mask[cs:c_out, :, :, :].zero_()
            for prev_mask in masks_c:
                mask -= prev_mask
            masks_c.append(mask)
        self.mask_c = TrainableMask(masks_c)

        # kernel masks
        masks_k = []
        for k in sorted(k_sizes):
            mask = torch.zeros(size=(1, 1, max_k, max_k), dtype=self.weight.dtype)
            dk = (max_k - k) // 2
            if dk == 0:
                mask += 1
            else:
                mask[:, :, dk:-dk, dk:-dk] += 1
            for prev_mask in masks_k:
                mask -= prev_mask
            masks_k.append(mask)
        self.mask_k = TrainableMask(masks_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(self.weight)
        mask = self.mask_c(self.weight, mask)
        mask = self.mask_k(self.weight, mask)
        weight = self.weight * mask
        return nn.functional.conv2d(x, weight, self.bias, padding=self._padding, stride=self._stride,
                                    dilation=self._dilation, groups=self._groups)

    def get_finalized_kernel(self) -> (int, int):
        """ get the finalized (idx, k_size) """
        idx = self.mask_k.get_finalized(self.weight)
        return idx, self.k_sizes[idx]

    def get_finalized_channel_mult(self) -> (int, float):
        """ get the finalized (idx, c_mul) """
        idx = self.mask_c.get_finalized(self.weight)
        return idx, self.c_multipliers[idx]


class SuperSqueezeExcitationChannelThresholdModule(nn.Module):
    def __init__(self, c: int, c_substitute: int = None, use_c_substitute: bool = False, divisible: int = None,
                 c_muls=(0.0, 0.25, 0.5), squeeze_act: str = 'relu', excite_act: str = 'sigmoid',
                 squeeze_bias=True, excite_bias=True, squeeze_bn=False, squeeze_bn_affine=False):
        """
        A squeeze and excitation module with searchable number of inner channels

        :param c: number of input and output channels
        :param c_substitute: used instead of 'c' for calculating inner channels, if not None and 'use_c_substitute'
        :param use_c_substitute: try using 'c_substitute'
        :param c_muls: tuple of multipliers for inner channels
        :param squeeze_act: activation function after squeezing
        :param excite_act: activation function after exciting
        :param squeeze_bias: use a bias for squeezing
        :param excite_bias: use a bias for exciting
        :param squeeze_bn: use a bn after squeezing
        :param squeeze_bn_affine: use an affine bn
        """
        super().__init__()
        self._kwargs = dict(att_cls='SqueezeExcitationChannelModule', use_c_substitute=use_c_substitute,
                            squeeze_act=squeeze_act, excite_act=excite_act,
                            squeeze_bias=squeeze_bias, excite_bias=excite_bias)
        self.c_muls = sorted(c_muls)
        cs = c_substitute if (isinstance(c_substitute, int) and use_c_substitute) else c
        max_c_mul = max(c_muls)
        c_muls_sub = [cm / max_c_mul for cm in self.c_muls]
        c_red = make_divisible(int(cs * max_c_mul), divisible)
        self.fc1 = SuperKernelThresholdConv(c, c_red, k_sizes=(1,), c_multipliers=c_muls_sub,
                                            groups=1, bias=squeeze_bias)
        ops = [
            nn.AdaptiveAvgPool2d(1),
            self.fc1
        ]
        if squeeze_bn:
            ops.append(nn.BatchNorm2d(c_red, affine=squeeze_bn_affine))
        ops.extend([
            Register.act_funs.get(squeeze_act)(inplace=True),
            nn.Conv2d(c_red, c, kernel_size=1, stride=1, bias=excite_bias),
            Register.act_funs.get(excite_act)(inplace=True),
        ])
        self.op = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.op(x)

    def att_kwargs(self):
        """ get the finalized att_dict """
        idx, _ = self.fc1.get_finalized_channel_mult()
        c_mul = self.c_muls[idx]
        if c_mul <= 0:
            return None
        kwargs = self._kwargs.copy()
        kwargs.update(dict(c_mul=c_mul))
        return kwargs


@Register.network_layer()
class SuperConvThresholdLayer(AbstractStepsLayer):
    def __init__(self, k_sizes=(3, 5, 7), dilation=1, stride=1, groups=1, bias=False, padding='same', **base_kwargs):
        """
        A regular convolution with a spatial mask for the kernel size

        :param k_sizes: kernel sizes for the spatial kernel
        :param dilation: dilation for the kernel
        :param stride: stride for the kernel
        :param padding: 'same' or number
        :param bias:
        :param padding:
        :param base_kwargs:
        """
        super().__init__(**base_kwargs)
        self._add_to_kwargs(k_sizes=k_sizes, dilation=dilation, stride=stride,
                            groups=groups, bias=bias, padding=padding)
        self.conv = None

    def _build(self, s_in: Shape, c_out: int, weight_functions=()) -> Shape:
        self.conv = SuperKernelThresholdConv(s_in.num_features(), c_out, self.k_sizes, (1.0,),
                                             self.dilation, self.stride, self.padding, self.groups, self.bias)
        wf = list(weight_functions) + [self.conv]
        return super()._build(s_in, c_out, weight_functions=wf)

    def config(self, finalize=False, **__):
        cfg = super().config(finalize=finalize, **__)
        if finalize:
            cfg['name'] = ConvLayer.__name__
            kwargs = cfg['kwargs']
            kwargs.pop('k_sizes')
            ks = self.conv.get_finalized_kernel()
            kwargs['k_size'] = ks[1]
            cfg['kwargs'] = kwargs
        return cfg


@Register.network_layer()
class SuperSepConvThresholdLayer(AbstractStepsLayer):
    def __init__(self, k_sizes=(3, 5, 7), dilation=1, stride=1, groups=-1, bias=False, padding='same', **base_kwargs):
        """
        A regular separable convolution with a spatial mask for the kernel size

        :param k_sizes: kernel sizes for the kernel
        :param dilation: dilation for the kernel
        :param stride: stride for the kernel
        :param padding: 'same' or number
        :param bias:
        :param padding:
        :param base_kwargs:
        """
        super().__init__(**base_kwargs)
        self._add_to_kwargs(k_sizes=k_sizes, dilation=dilation, stride=stride,
                            groups=groups, bias=bias, padding=padding)
        self.conv = None

    def _build(self, s_in: Shape, c_out: int, weight_functions=()) -> Shape:
        self.conv = SuperKernelThresholdConv(s_in.num_features(), s_in.num_features(), self.k_sizes, (1.0,),
                                             self.dilation, self.stride, self.padding, self.groups, self.bias)
        point_conv = nn.Conv2d(s_in.num_features(), c_out, kernel_size=1, groups=1, bias=self.bias)
        wf = list(weight_functions) + [self.conv, point_conv]
        return super()._build(s_in, c_out, weight_functions=wf)

    def config(self, finalize=False, **__):
        cfg = super().config(finalize=finalize, **__)
        if finalize:
            cfg['name'] = SepConvLayer.__name__
            kwargs = cfg['kwargs']
            kwargs.pop('k_sizes')
            ks = self.conv.get_finalized_kernel()
            kwargs['k_size'] = ks[1]
            cfg['kwargs'] = kwargs
        return cfg


@Register.network_layer()
class SuperMobileInvertedConvThresholdLayer(AbstractLayer):
    def __init__(self,  k_sizes=(3, 5, 7), stride=1, padding='same', expansions=(3, 6),
                 dilation=1, bn_affine=True, act_fun='relu6', act_inplace=True, sse_dict=None):
        """
        A super kernel layer for several kernel sizes and expansion sizes, to share as many weights as possible.

        :param k_sizes: kernel sizes for the spatial kernel
        :param stride: stride for the spatial kernel
        :param padding: 'same' or number
        :param expansions: multipliers for inner channels, based on input channels
        :param dilation: dilation for the spatial kernel
        :param bn_affine: affine batch norm
        :param act_fun: activation function
        :param act_inplace: whether to use the activation function in-place if possible (e.g. ReLU)
        :param sse_dict: None to disable squeeze+excitation, otherwise a dict with respective kwargs
        """
        super().__init__()
        self._add_to_kwargs(k_sizes=k_sizes, stride=stride, expansions=sorted(expansions),
                            padding=padding, dilation=dilation,
                            bn_affine=bn_affine, act_fun=act_fun, act_inplace=act_inplace, sse_dict=sse_dict)
        self._add_to_print_kwargs(has_skip=False)
        self.conv = None
        self.block = None

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        c_in = s_in.num_features()
        c_mid = int(c_in * max(self.expansions))
        self.has_skip = self.stride == 1 and c_in == c_out
        max_exp = max(self.expansions)
        exp_mults = [e / max_exp for e in self.expansions]
        ops = []

        if max_exp > 1:
            # pw
            ops.extend([
                nn.Conv2d(c_in, c_mid, 1, 1, 0, groups=1, bias=False),
                nn.BatchNorm2d(c_mid, affine=self.bn_affine),
                Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            ])
        # dw
        self.conv = SuperKernelThresholdConv(c_mid, c_mid, self.k_sizes, exp_mults, self.dilation, self.stride,
                                             self.padding, -1, bias=False)
        ops.extend([
            self.conv,
            nn.BatchNorm2d(c_mid, affine=self.bn_affine),
            Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
        ])
        # optional squeeze+excitation module with searchable width
        if isinstance(self.sse_dict, dict):
            self.learned_se = SuperSqueezeExcitationChannelThresholdModule(c_mid, c_substitute=c_in, **self.sse_dict)
            ops.append(self.learned_se)
        else:
            self.learned_se = None
        # pw
        ops.extend([
            nn.Conv2d(c_mid, c_out, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(c_out, affine=self.bn_affine),
        ])
        self.block = nn.Sequential(*ops)
        if self.has_skip:
            self.block = DropPathModule(self.block)
        return self.probe_outputs(s_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_skip:
            return x + self.block(x)
        return self.block(x)

    def config(self, finalize=False, **__):
        cfg = super().config(finalize=finalize, **__)
        if finalize:
            cfg['name'] = MobileInvertedConvLayer.__name__
            kwargs = cfg['kwargs']
            for s in ['k_sizes', 'expansions', 'sse_dict']:
                kwargs.pop(s)
            ks = self.conv.get_finalized_kernel()
            es = self.conv.get_finalized_channel_mult()
            kwargs['k_size'] = ks[1]
            kwargs['expansion'] = self.expansions[es[0]]
            kwargs['att_dict'] = None if self.learned_se is None else self.learned_se.att_kwargs()
            cfg['kwargs'] = kwargs
        return cfg
