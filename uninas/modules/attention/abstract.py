import torch
import torch.nn as nn
from uninas.register import Register
from uninas.modules.layers.abstract import AbstractLayer
from uninas.utils.shape import Shape


class AbstractAttentionModule(nn.Module):
    def __init__(self, c: int, c_substitute: int = None, use_c_substitute=False):
        """

        :param c: number of input and output channels
        :param c_substitute: used instead of 'c' for calculating inner channels, if not None and 'use_c_substitute'
                             in MobileNet and ShuffleNet blocks this is the number of block input channels
                             (usually fewer than the input channels of the SE module within the block)
        :param use_c_substitute: try using 'c_substitute'

        c and c_substitute are given by the block using an attention layer (e.g. MobileNetV2),
        all other parameters are usually given via primitives
        """
        super().__init__()
        self.c = c_substitute if (isinstance(c_substitute, int) and use_c_substitute) else c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def module_from_dict(cls, c: int, c_substitute: int = None, att_dict: dict = None) -> nn.Module:
        assert isinstance(att_dict, dict), "att_dict is not a dict"
        att_dict = att_dict.copy()
        att_cls = Register.attention_modules.get(att_dict.pop('att_cls'))
        return att_cls(c, c_substitute, **att_dict)


@Register.network_layer()
class AttentionLayer(AbstractLayer):

    def __init__(self, att_dict: dict, **__):
        super().__init__(**__)
        self._add_to_kwargs(att_dict=att_dict)
        self.att_module = None

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        assert s_in.num_features() == c_out
        self.att_module = AbstractAttentionModule.module_from_dict(c_out, c_substitute=None, att_dict=self.att_dict)
        return s_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.att_module(x)
