"""
ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
https://arxiv.org/abs/1807.11164
"""

import torch
import torch.nn as nn
from uninas.modules.layers.abstract import AbstractLayer
from uninas.modules.attention.abstract import AbstractAttentionModule
from uninas.modules.modules.misc import DropPathModule
from uninas.utils.torch.misc import get_padding
from uninas.utils.shape import Shape
from uninas.register import Register


class ShuffleChannelModule(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleChannelModule, self).__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.data.size()
        channels_per_group = c // self.groups
        x = x.view(n, self.groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(n, c, h, w)


class AbstractShuffleNetLayer(AbstractLayer):
    def __init__(self, k_size=3, stride=1, padding='same', expansion=1.0, dilation=1, bn_affine=False,
                 act_fun='relu', act_inplace=False, att_dict: dict = None):
        """

        :param k_size: kernel size for the spatial kernel
        :param stride: stride for the spatial kernel
        :param padding: 'same' or number
        :param expansion: multiplier for inner channels, based on input channels
        :param dilation: dilation for the spatial kernel
        :param bn_affine: affine batch norm
        :param act_fun: activation function
        :param act_inplace: whether to use the activation function in-place if possible (e.g. ReLU)
                            conflicts with DropPathModule in this type of layer
        :param att_dict: None to disable attention modules, otherwise a dict with respective kwargs
        """
        super().__init__()
        self._add_to_kwargs(k_size=k_size, stride=stride, expansion=expansion, padding=padding, dilation=dilation,
                            bn_affine=bn_affine, act_fun=act_fun, act_inplace=act_inplace, att_dict=att_dict)
        self.branch_main = None
        self.branch_proj = None

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x_proj, x_main = self.channel_shuffle(x)
            return torch.cat((x_proj, self.branch_main(x_main)), dim=1)
        elif self.stride == 2:
            return torch.cat((self.branch_proj(x), self.branch_main(x)), dim=1)

    @staticmethod
    def channel_shuffle(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        n, c, h, w = x.data.size()
        x = x.reshape(n * c // 2, 2, h * w)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, c // 2, h, w)
        return x[0], x[1]


@Register.network_layer()
class ShuffleNetV2Layer(AbstractShuffleNetLayer):
    def _build(self, s_in: Shape, c_out: int) -> Shape:
        assert not (c_out <= s_in.num_features() and self.stride > 1), "must increase num features when stride is >1"
        assert s_in.num_features() % 4 == 0 and c_out % 2 == 0, "num features must be divisible by 4"

        padding = get_padding(self.padding, self.k_size, self.stride, self.dilation)

        if self.stride >= 2:
            c_side = c_main_in = s_in.num_features()

            self.branch_proj = nn.Sequential(*[
                # dw
                nn.Conv2d(c_side, c_side, self.k_size, self.stride, padding, groups=c_side, bias=False),
                nn.BatchNorm2d(c_side, affine=self.bn_affine),
                # pw
                nn.Conv2d(c_side, c_side, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c_side, affine=self.bn_affine),
                Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            ])
        else:
            c_side = c_main_in = s_in.num_features() // 2
        c_main_out = c_out - c_side
        c_main_mid = int(c_out // 2 * self.expansion)

        bm = [
            # pw
            nn.Conv2d(c_main_in, c_main_mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_main_mid, affine=self.bn_affine),
            Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            # dw
            nn.Conv2d(c_main_mid, c_main_mid, self.k_size, self.stride, padding, groups=c_main_mid, bias=False),
            nn.BatchNorm2d(c_main_mid, affine=self.bn_affine),
            # pw
            nn.Conv2d(c_main_mid, c_main_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_main_out, affine=self.bn_affine),
            Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
        ]
        # optional attention module
        if isinstance(self.att_dict, dict):
            bm.append(AbstractAttentionModule.module_from_dict(c_main_out, c_substitute=c_main_in,
                                                               att_dict=self.att_dict))

        # self.branch_main = nn.Sequential(*bm)
        self.branch_main = DropPathModule(nn.Sequential(*bm))
        return self.probe_outputs(s_in)


@Register.network_layer()
class ShuffleNetV2XceptionLayer(AbstractShuffleNetLayer):
    def _build(self, s_in: Shape, c_out: int) -> Shape:
        assert not (c_out <= s_in.num_features() and self.stride > 1), "must increase num features when stride is >1"
        assert s_in.num_features() % 4 == 0 and c_out % 2 == 0, "num features must be divisible by 4"

        padding = get_padding(self.padding, self.k_size, self.stride, self.dilation)
        padding2 = get_padding(self.padding, self.k_size, 1, self.dilation)

        if self.stride >= 2:
            c_side = c_main_in = s_in.num_features()

            self.branch_proj = nn.Sequential(*[
                # dw
                nn.Conv2d(c_side, c_side, self.k_size, self.stride, padding, groups=c_side, bias=False),
                nn.BatchNorm2d(c_side, affine=self.bn_affine),
                # pw
                nn.Conv2d(c_side, c_side, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c_side, affine=self.bn_affine),
                Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            ])
        else:
            c_side = c_main_in = s_in.num_features() // 2
        c_main_out = c_out - c_side
        c_main_mid = int(c_out // 2 * self.expansion)

        bm = [
            # dw 1
            nn.Conv2d(c_main_in, c_main_in, self.k_size, self.stride, padding, groups=c_main_in, bias=False),
            nn.BatchNorm2d(c_main_in, affine=self.bn_affine),
            # pw 1
            nn.Conv2d(c_main_in, c_main_mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_main_mid, affine=self.bn_affine),
            Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            # dw 2
            nn.Conv2d(c_main_mid, c_main_mid, self.k_size, 1, padding2, groups=c_main_mid, bias=False),
            nn.BatchNorm2d(c_main_mid, affine=self.bn_affine),
            # pw 2
            nn.Conv2d(c_main_mid, c_main_mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_main_mid, affine=self.bn_affine),
            Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
            # dw 3
            nn.Conv2d(c_main_mid, c_main_mid, self.k_size, 1, padding2, groups=c_main_mid, bias=False),
            nn.BatchNorm2d(c_main_mid, affine=self.bn_affine),
            # pw 3
            nn.Conv2d(c_main_mid, c_main_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_main_out, affine=self.bn_affine),
            Register.act_funs.get(self.act_fun)(inplace=self.act_inplace),
        ]
        # optional attention module
        if isinstance(self.att_dict, dict):
            bm.append(AbstractAttentionModule.module_from_dict(c_main_out, c_substitute=c_main_in,
                                                               att_dict=self.att_dict))

        # self.branch_main = nn.Sequential(*bm)
        self.branch_main = DropPathModule(nn.Sequential(*bm))
        return self.probe_outputs(s_in)
