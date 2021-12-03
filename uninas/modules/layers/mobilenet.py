"""
MobileNetV2: Inverted Residuals and Linear Bottlenecks
https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
from uninas.modules.layers.abstract import AbstractLayer
from uninas.modules.layers.mixconv import MixConvModule
from uninas.modules.attention.abstract import AbstractAttentionModule
from uninas.modules.modules.shared import AbstractSharedPathsOp
from uninas.modules.modules.misc import DropPathModule
from uninas.utils.torch.misc import get_padding, make_divisible
from uninas.utils.shape import Shape
from uninas.register import Register


def get_conv2d(c_in: int, c_out: int, k_size, stride=1, groups=-1, dilation=1, padding='same') -> nn.Module:
    # multiple kernel sizes, mix conv
    if isinstance(k_size, (tuple, list)):
        if len(k_size) > 1:
            return MixConvModule(c_in, c_out, k_size=k_size, stride=stride,
                                 dilation=dilation, padding=padding, groups=groups, bias=False,
                                 mode='even', divisible=1)
        k_size = k_size[0]
    # one kernel size, regular conv
    padding = get_padding(padding, k_size, stride, dilation)
    groups = c_in if groups == -1 else groups
    return nn.Conv2d(c_in, c_out, k_size, stride, padding, groups=groups, bias=False)


@Register.network_layer()
class MobileInvertedConvLayer(AbstractLayer):

    def __init__(self, k_size=3, k_size_in=1, k_size_out=1, stride=1, padding='same', expansion=6, dilation=1,
                 bn_affine=True, act_fun='relu6', act_inplace=True, att_dict: dict = None, fused=False):
        """

        :param k_size: kernel size(s) for the spatial kernel
        :param k_size_in: kernel size(s) for the first conv kernel (expanding)
        :param k_size_out: kernel size(s) for the last conv kernel (projecting)
        :param stride: stride for the spatial kernel
        :param padding: 'same' or number
        :param expansion: multiplier for inner channels, based on input channels
        :param dilation: dilation for the spatial kernel
        :param bn_affine: affine batch norm
        :param act_fun: activation function
        :param act_inplace: whether to use the activation function in-place if possible (e.g. ReLU)
        :param att_dict: None to disable attention modules, otherwise a dict with respective kwargs
        :param fused: fuse the initial pointwise and depthwise convolutions
        """
        super().__init__()
        self._add_to_kwargs(k_size=k_size, k_size_in=k_size_in, k_size_out=k_size_out, stride=stride,
                            expansion=expansion, padding=padding, dilation=dilation, bn_affine=bn_affine,
                            act_fun=act_fun, act_inplace=act_inplace, att_dict=att_dict, fused=fused)
        self._add_to_print_kwargs(has_skip=False)
        self.block = None

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        c_in = s_in.num_features()
        c_mid = make_divisible(int(c_in * self.expansion), divisible=8)
        self.has_skip = self.stride == 1 and c_in == c_out
        ops = []
        conv_kwargs = dict(dilation=self.dilation, padding=self.padding)

        # fused?
        if not self.fused:
            if self.expansion > 1:
                # pw
                ops.extend([
                    get_conv2d(c_in, c_mid, k_size=self.k_size_in, groups=1, **conv_kwargs),
                    nn.BatchNorm2d(c_mid, affine=self.bn_affine),
                    Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
                ])
            # dw
            ops.extend([
                get_conv2d(c_mid, c_mid, k_size=self.k_size, stride=self.stride, groups=-1, **conv_kwargs),
                nn.BatchNorm2d(c_mid, affine=self.bn_affine),
                Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            ])
        else:
            ops.extend([
                get_conv2d(c_in, c_mid, k_size=self.k_size, stride=self.stride, groups=1, **conv_kwargs),
                nn.BatchNorm2d(c_mid, affine=self.bn_affine),
                Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            ])
        # optional squeeze+excitation module
        if isinstance(self.att_dict, dict):
            ops.append(AbstractAttentionModule.module_from_dict(c_mid, c_substitute=c_in, att_dict=self.att_dict))
        # final pw
        ops.extend([
            get_conv2d(c_mid, c_out, k_size=self.k_size_out, groups=1, **conv_kwargs),
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


@Register.network_layer()
class SharedMixedMobileInvertedConvLayer(AbstractSharedPathsOp):

    def __init__(self, name: str, strategy_name='default', skip_op: str = None, k_size_in=1, k_size_out=1,
                 k_sizes=(3, 5, 7), stride=1, padding='same', expansions=(3, 6),
                 dilation=1, bn_affine=True, act_fun='relu6', act_inplace=True, att_dict: dict = None, fused=False):
        """
        A layer for several kernel sizes and expansion sizes sharing the 1x1 conv weights.
        Currently only designed for having a single kernel+expansion per forward pass and for the final config.

        :param name: name under which to register architecture weights
        :param strategy_name: name of the strategy for architecture weights
        :param skip_op: optional layer name, adds an op that enables skipping the entire block, e.g. "SkipLayer"
        :param k_size_in: kernel size(s) for the first conv kernel (expanding)
        :param k_size_out: kernel size(s) for the last conv kernel (projecting)
        :param k_sizes: kernel sizes for the spatial kernel
        :param stride: stride for the spatial kernel
        :param padding: 'same' or number
        :param expansions: multipliers for inner channels, based on input channels
        :param dilation: dilation for the spatial kernel
        :param bn_affine: affine batch norm
        :param act_fun: activation function
        :param act_inplace: whether to use the activation function in-place if possible (e.g. ReLU)
        :param att_dict: None to disable attention modules, otherwise a dict with respective kwargs
        :param fused: fuse the initial pointwise and depthwise convolutions
        """
        super().__init__(name, strategy_name)
        self._add_to_kwargs(skip_op=skip_op, k_size_in=k_size_in, k_size_out=k_size_out,
                            k_sizes=k_sizes, stride=stride, expansions=expansions,
                            padding=padding, dilation=dilation, bn_affine=bn_affine,
                            act_fun=act_fun, act_inplace=act_inplace, att_dict=att_dict, fused=fused)
        self._add_to_print_kwargs(has_skip=False, has_att=isinstance(self.att_dict, dict))
        self.pw_in = nn.ModuleList([])
        self.dw_conv = nn.ModuleList([])
        self.dw_att = nn.ModuleList([])
        self.pw_out = nn.ModuleList([])
        self.drop_path = DropPathModule()
        self.skip = None
        self.has_skip = self.stride == 1 and isinstance(self.skip_op, str)

        # all paths, using shared weights
        for e in range(len(self.expansions)):
            for k in range(len(self.k_sizes)):
                self._add_shared_path((e, k))
        if self.has_skip:
            self._add_shared_path(('skip', 'skip'))

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        conv_kwargs = dict(dilation=self.dilation, padding=self.padding)
        c_in = s_in.num_features()
        if self.has_skip:
            assert c_in == c_out, "If a skip function is used, the same input/output channel counts are required"
            self.skip = Register.network_layers.get(self.skip_op)()
            self.skip.build(s_in, c_out)

        for e in self.expansions:
            convs = nn.ModuleList([])
            c_mid = int(c_in * e)

            if self.fused:
                # substitute pw_in with identity, use fused conv as dw
                self.pw_in.append(nn.Identity())
                for k in self.k_sizes:
                    convs.append(nn.Sequential(
                        get_conv2d(c_in, c_mid, k_size=k, stride=self.stride, groups=1, **conv_kwargs),
                        nn.BatchNorm2d(c_mid, affine=self.bn_affine),
                        Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
                    ))
            else:
                # pw in
                self.pw_in.append(nn.Sequential(
                    get_conv2d(c_in, c_mid, k_size=self.k_size_in, groups=1, **conv_kwargs),
                    nn.BatchNorm2d(c_mid, affine=self.bn_affine),
                    Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
                ))
                # dw conv ops with different kernel sizes
                for k in self.k_sizes:
                    convs.append(nn.Sequential(
                        get_conv2d(c_mid, c_mid, k_size=k, stride=self.stride, groups=-1, **conv_kwargs),
                        nn.BatchNorm2d(c_mid, affine=self.bn_affine),
                        Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
                    ))
            self.dw_conv.append(convs)
            # dw optional attention module
            if self.has_att:
                self.dw_att.append(AbstractAttentionModule.module_from_dict(c_mid, c_substitute=c_in,
                                                                            att_dict=self.att_dict))
            # pw out
            self.pw_out.append(nn.Sequential(
                get_conv2d(c_mid, c_out, k_size=self.k_size_out, groups=1, **conv_kwargs),
                nn.BatchNorm2d(c_out, affine=self.bn_affine),
            ))
        return self.probe_outputs(s_in)

    def forward_path(self, x: torch.Tensor, path: tuple) -> torch.Tensor:
        idx_e, idx_k = path
        if idx_e == 'skip':
            return x + self.skip(x)
        x2 = self.pw_in[idx_e](x)
        x2 = self.dw_conv[idx_e][idx_k](x2)
        if self.has_att:
            x2 = self.dw_att[idx_e](x2)
        x2 = self.pw_out[idx_e](x2)
        if self.has_skip:
            return x + self.drop_path(x2)
        return x2

    def config_path(self, cfg: dict, path: tuple, finalize=False, **__) -> dict:
        if finalize:
            idx_e, idx_k = path
            if idx_e == 'skip':
                return self.skip.config(finalize=finalize, **__)
            cfg['name'] = MobileInvertedConvLayer.__name__
            kwargs = cfg['kwargs']
            for s in ['name', 'strategy_name', 'skip_op']:
                kwargs.pop(s)
            kwargs['k_size'] = kwargs.pop('k_sizes')[idx_k]
            kwargs['expansion'] = kwargs.pop('expansions')[idx_e]
            cfg['kwargs'] = kwargs
        return cfg
