"""
MobileNetV2: Inverted Residuals and Linear Bottlenecks
https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
from uninas.methods.strategies.manager import StrategyManager
from uninas.model.layers.abstract import AbstractLayer
from uninas.model.layers.mixconv import MixConvModule
from uninas.model.attention.abstract import AbstractAttentionModule
from uninas.model.modules.fused import FusedOp
from uninas.model.modules.misc import DropPathModule
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
                 bn_affine=True, act_fun='relu6', act_inplace=True, att_dict: dict = None):
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
        """
        super().__init__()
        self._add_to_kwargs(k_size=k_size, k_size_in=k_size_in, k_size_out=k_size_out, stride=stride,
                            expansion=expansion, padding=padding, dilation=dilation, bn_affine=bn_affine,
                            act_fun=act_fun, act_inplace=act_inplace, att_dict=att_dict)
        self._add_to_print_kwargs(has_skip=False)
        self.block = None

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        c_in = s_in.num_features()
        c_mid = make_divisible(int(c_in * self.expansion), divisible=8)
        self.has_skip = self.stride == 1 and c_in == c_out
        ops = []
        conv_kwargs = dict(dilation=self.dilation, padding=self.padding)

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
        # optional squeeze+excitation module
        if isinstance(self.att_dict, dict):
            ops.append(AbstractAttentionModule.module_from_dict(c_mid, c_substitute=c_in, att_dict=self.att_dict))
        # pw
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
class FusedMobileInvertedConvLayer(AbstractLayer, FusedOp):

    def __init__(self, name: str, strategy_name='default', skip_op: str = None, k_size_in=1, k_size_out=1,
                 k_sizes=(3, 5, 7), stride=1, padding='same', expansions=(3, 6),
                 dilation=1, bn_affine=True, act_fun='relu6', act_inplace=True, att_dict: dict = None):
        """
        A fused layer for several kernel sizes and expansion sizes, to share the 1x1 conv weights.
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
        """
        super().__init__()
        self._add_to_kwargs(name=name, strategy_name=strategy_name, skip_op=skip_op,
                            k_size_in=k_size_in, k_size_out=k_size_out,
                            k_sizes=k_sizes, stride=stride, expansions=expansions,
                            padding=padding, dilation=dilation, bn_affine=bn_affine,
                            act_fun=act_fun, act_inplace=act_inplace, att_dict=att_dict)
        self._add_to_print_kwargs(has_skip=False, has_att=isinstance(self.att_dict, dict))
        self.ws = None
        self.skip = None
        self.pw_in = nn.ModuleList([])
        self.dw_conv = nn.ModuleList([])
        self.dw_att = nn.ModuleList([])
        self.pw_out = nn.ModuleList([])
        self.drop_path = DropPathModule()

        self._choices_by_idx = []

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        conv_kwargs = dict(dilation=self.dilation, padding=self.padding)
        c_in = s_in.num_features()
        self.has_skip = self.stride == 1 and c_in == c_out
        for e in range(len(self.expansions)):
            for k in range(len(self.k_sizes)):
                self._choices_by_idx.append((e, k))
        if self.has_skip and isinstance(self.skip_op, str):
            self.skip = Register.network_layers.get(self.skip_op)()
            self.skip.build(s_in, c_out)
            self._choices_by_idx.append(('skip', 'skip'))
        self.ws = StrategyManager().make_weight(self.strategy_name, self.name, only_single_path=True,
                                                num_choices=len(self._choices_by_idx))

        for e in self.expansions:
            c_mid = int(c_in * e)
            # pw in
            self.pw_in.append(nn.Sequential(
                get_conv2d(c_in, c_mid, k_size=self.k_size_in, groups=1, **conv_kwargs),
                nn.BatchNorm2d(c_mid, affine=self.bn_affine),
                Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            ))
            # dw conv ops with different kernel sizes
            convs = nn.ModuleList([])
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx, _ = self.ws.combine_info(self.name)[0]
        idx_e, idx_k = self._choices_by_idx[idx]
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

    def config(self, finalize=False, **__) -> dict:
        cfg = super().config(finalize=finalize, **__)
        if finalize:
            idxs = self.ws.get_finalized_indices(self.name)
            assert len(idxs) == 1
            idx_e, idx_k = self._choices_by_idx[idxs[0]]
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
