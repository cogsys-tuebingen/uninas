"""
Convolutions and MobileNet blocks that make heavy use of weight sharing.
Contrary to  Single-Path NAS, architecture parameters are required (but enable a super2 search)
"""

import torch
import torch.nn as nn
from uninas.methods.strategy_manager import StrategyManager
from uninas.modules.layers.abstract import AbstractLayer, AbstractStepsLayer
from uninas.modules.layers.cnn import ConvLayer, SepConvLayer
from uninas.modules.layers.mobilenet import MobileInvertedConvLayer
from uninas.modules.attention.abstract import AbstractAttentionModule
from uninas.modules.modules.misc import DropPathModule
from uninas.utils.torch.misc import get_padding, make_divisible
from uninas.utils.shape import Shape
from uninas.utils.misc import get_number
from uninas.register import Register


class SuperKernelConv(nn.Module):

    def __init__(self, c_in: int, c_out: int, name: str, strategy_name='default', k_sizes=(3, 5),
                 c_multipliers=(0.5, 1.0), dilation=1, stride=1, padding='same', groups=-1, bias=False):
        """
        A super-kernel that applies convolution with a masked weight, using architecture weights to figure out the best
        masking, thus kernel size and num output channels. Since the architecture weights are applied to the mask rather
        than generating different outputs, this module can be used efficiently for differentiable weight strategies.

        :param c_in: num input channels
        :param c_out: num output channels
        :param name: name under which to register architecture weights
        :param strategy_name: name of the strategy for architecture weights
        :param k_sizes: kernel sizes
        :param c_multipliers:
        :param dilation: dilation for the kernel
        :param stride: stride for the kernel
        :param padding:
        :param padding: 'same' or number
        :param bias:
        """
        super().__init__()
        self.name_c = '%s/c' % name
        self.name_k = '%s/k' % name
        self.k_sizes = k_sizes
        self.c_multipliers = c_multipliers
        assert max(c_multipliers) <= 1.0, "Can only reduce max channels, choose a higher c_in/c_out"

        self._stride = stride
        self._groups = get_number(groups, c_out)
        self._dilation = dilation
        assert c_in % self._groups == 0

        max_k = max(k_sizes)
        channels = [int(c_out * ci) for ci in sorted(c_multipliers)]
        masks_c, masks_k = [], []

        # arc weights
        self.ws = StrategyManager().make_weight(strategy_name, self.name_k, only_single_path=True, num_choices=len(k_sizes))
        self.ws = StrategyManager().make_weight(strategy_name, self.name_c, only_single_path=True, num_choices=len(channels))

        # conv weight
        self._padding = get_padding(padding, max_k, stride, 1)
        self.weight = nn.Parameter(torch.Tensor(c_out, c_in // self._groups, max_k, max_k), requires_grad=True)
        nn.init.kaiming_normal_(self.weight, mode='fan_out')

        # bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(c_out))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

        # mask c
        for cs in channels:
            mask = torch.ones(size=(c_out, 1, 1, 1), dtype=self.weight.dtype)
            mask[cs:c_out, :, :, :].zero_()
            masks_c.append(mask)
        self.register_buffer('masks_c', torch.stack(masks_c, dim=0))

        # mask k
        for k in sorted(k_sizes):
            mask = torch.zeros(size=(1, 1, max_k, max_k), dtype=self.weight.dtype)
            dk = (max_k - k) // 2
            if dk == 0:
                mask += 1
            else:
                mask[:, :, dk:-dk, dk:-dk] += 1
            masks_k.append(mask)
        self.register_buffer('masks_k', torch.stack(masks_k, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        mask_c = sum([self.masks_c[ic]*iw for ic, iw in self.ws.combine_info(self.name_c)])
        mask_k = sum([self.masks_k[ik]*iw for ik, iw in self.ws.combine_info(self.name_k)])

        return nn.functional.conv2d(x, weight*mask_c*mask_k, self.bias, self._stride, self._padding,
                                    self._dilation, self._groups)

    def get_finalized_kernel(self) -> [(int, int)]:
        """ get the list of finalized (idx, k_size) """
        ik = self.ws.get_finalized_index(self.name_k)
        return ik, self.k_sizes[ik]

    def get_finalized_channel_mult(self) -> [(int, int)]:
        """ get the finalized (idx, c_mul) """
        ic = self.ws.get_finalized_index(self.name_c)
        return ic, self.c_multipliers[ic]


@Register.network_layer()
class SuperConvLayer(AbstractStepsLayer):

    def __init__(self, name: str, strategy_name='default', k_sizes=(3, 5, 7), dilation=1, stride=1, groups=1,
                 bias=False, padding='same', **base_kwargs):
        """
        A regular convolution with a spatial mask for the kernel size

        :param name: name under which to register architecture weights
        :param strategy_name: name of the strategy for architecture weights
        :param k_sizes: kernel sizes for the spatial kernel
        :param dilation: dilation for the kernel
        :param stride: stride for the kernel
        :param padding: 'same' or number
        :param bias:
        :param padding:
        :param base_kwargs:
        """
        super().__init__(**base_kwargs)
        self._add_to_kwargs(name=name, strategy_name=strategy_name, k_sizes=k_sizes, dilation=dilation, stride=stride,
                            groups=groups, bias=bias, padding=padding)
        self.conv = None

    def _build(self, s_in: Shape, c_out: int, weight_functions=()) -> Shape:
        self.conv = SuperKernelConv(s_in.num_features(), c_out, self.name, self.strategy_name, self.k_sizes, (1.0,),
                                    self.dilation, self.stride, self.padding, self.groups, self.bias)
        wf = list(weight_functions) + [self.conv]
        return super()._build(s_in, c_out, weight_functions=wf)

    def config(self, finalize=False, **__):
        cfg = super().config(finalize=finalize, **__)
        if finalize:
            cfg['name'] = ConvLayer.__name__
            kwargs = cfg['kwargs']
            kwargs.pop('name')
            kwargs.pop('strategy_name')
            kwargs.pop('k_sizes')
            ks = self.conv.get_finalized_kernel()
            kwargs['k_size'] = ks[1]
            cfg['kwargs'] = kwargs
        return cfg


@Register.network_layer()
class SuperSepConvLayer(AbstractStepsLayer):

    def __init__(self, name: str, strategy_name='default', k_sizes=(3, 5, 7), dilation=1, stride=1, groups=1,
                 bias=False, padding='same', **base_kwargs):
        """
        A regular separable convolution with a spatial mask for the kernel size

        :param name: name under which to register architecture weights
        :param strategy_name: name of the strategy for architecture weights
        :param k_sizes: kernel sizes for the kernel
        :param dilation: dilation for the kernel
        :param stride: stride for the kernel
        :param padding: 'same' or number
        :param bias:
        :param padding:
        :param base_kwargs:
        """
        super().__init__(**base_kwargs)
        self._add_to_kwargs(name=name, strategy_name=strategy_name, k_sizes=k_sizes, dilation=dilation, stride=stride,
                            groups=groups, bias=bias, padding=padding)
        self.conv = None

    def _build(self, s_in: Shape, c_out: int, weight_functions=()) -> Shape:
        c_in = s_in.num_features()
        self.conv = SuperKernelConv(c_in, c_in, self.name, self.strategy_name, self.k_sizes, (1.0,),
                                    self.dilation, self.stride, self.padding, self.groups, self.bias)
        point_conv = nn.Conv2d(c_in, c_out, kernel_size=1,
                               groups=get_number(self.groups, s_in.num_features()), bias=self.bias)
        wf = list(weight_functions) + [self.conv, point_conv]
        return super()._build(s_in, c_out, weight_functions=wf)

    def config(self, finalize=False, **__):
        cfg = super().config(finalize=finalize, **__)
        if finalize:
            cfg['name'] = SepConvLayer.__name__
            kwargs = cfg['kwargs']
            kwargs.pop('name')
            kwargs.pop('strategy_name')
            kwargs.pop('k_sizes')
            ks = self.conv.get_finalized_kernel()
            kwargs['k_size'] = ks[1]
            cfg['kwargs'] = kwargs
        return cfg


@Register.network_layer()
class SuperMobileInvertedConvLayer(AbstractLayer):

    def __init__(self, name: str, strategy_name='default',
                 k_sizes=(3, 5, 7), stride=1, padding='same', expansions=(3, 6), dilation=1, bn_affine=True,
                 act_fun='relu6', act_inplace=True, att_dict: dict = None):
        """
        A super kernel layer for several kernel sizes and expansion sizes, to share as many weights as possible.

        :param name: name under which to register architecture weights
        :param strategy_name: name of the strategy for architecture weights
        :param k_sizes: kernel sizes for the spatial kernel
        :param stride: stride for the spatial kernel
        :param padding: 'same' or number
        :param expansions: multipliers for inner channels, based on input channels
        :param dilation: dilation for the spatial kernel
        :param bn_affine: affine batch norm
        :param act_fun: activation function
        :param act_inplace: whether to use the activation function in-place if possible (e.g. ReLU)
        :param att_dict: None to disable attention modules, otherwise a dict with respective kwargs
        """
        super().__init__()
        self._add_to_kwargs(name=name, strategy_name=strategy_name,
                            k_sizes=k_sizes, stride=stride, expansions=sorted(expansions), padding=padding,
                            dilation=dilation, bn_affine=bn_affine, act_fun=act_fun, act_inplace=act_inplace,
                            att_dict=att_dict)
        self._add_to_print_kwargs(has_skip=False)
        self.conv = None
        self.block = None

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        c_in = s_in.num_features()
        max_exp = max(self.expansions)
        exp_mults = [e / max_exp for e in self.expansions]
        c_mid = make_divisible(int(c_in * max_exp), divisible=8)
        self.has_skip = self.stride == 1 and c_in == c_out
        ops = []

        self.conv = SuperKernelConv(c_mid, c_mid, self.name, self.strategy_name, self.k_sizes, exp_mults,
                                    self.dilation, self.stride, self.padding, c_mid, bias=False)

        if max_exp > 1:
            # pw
            ops.extend([
                nn.Conv2d(c_in, c_mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c_mid, affine=self.bn_affine),
                Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            ])
        # dw
        ops.extend([
            self.conv,
            nn.BatchNorm2d(c_mid, affine=self.bn_affine),
            Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
        ])
        # optional attention module
        if isinstance(self.att_dict, dict):
            ops.append(AbstractAttentionModule.module_from_dict(c_mid, c_substitute=c_in, **self.att_dict))
        # pw
        ops.extend([
            nn.Conv2d(c_mid, c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_out, affine=self.bn_affine),
        ])
        self.block = nn.Sequential(*ops)
        if self.has_skip:
            self.block = DropPathModule(self.block)
        return self.probe_outputs(s_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_skip:
            return x + self.block.forward(x)
        return self.block.forward(x)

    def config(self, finalize=False, **__):
        cfg = super().config(finalize=finalize, **__)
        if finalize:
            cfg['name'] = MobileInvertedConvLayer.__name__
            kwargs = cfg['kwargs']
            kwargs.pop('name')
            kwargs.pop('strategy_name')
            kwargs.pop('k_sizes')
            kwargs.pop('expansions')
            ks = self.conv.get_finalized_kernel()
            es = self.conv.get_finalized_channel_mult()
            kwargs['k_size'] = ks[1]
            kwargs['expansion'] = self.expansions[es[0]]
            cfg['kwargs'] = kwargs
        return cfg
