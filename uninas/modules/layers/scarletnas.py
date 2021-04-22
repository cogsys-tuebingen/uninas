"""
SCARLET-NAS: Bridging the gap between Stability and Scalability in Weight-sharing Neural Architecture Search
https://arxiv.org/abs/1908.06022
"""

import torch
import torch.nn as nn
from uninas.modules.layers.abstract import AbstractLayer
from uninas.modules.layers.common import SkipLayer
from uninas.register import Register
from uninas.utils.shape import Shape


@Register.network_layer()
class LinearTransformerLayer(AbstractLayer):
    """
    A linear transformer layer, proposed in ScarletNAS
    Is turned into a SkipLayer when finalized
    """

    def __init__(self, **base_kwargs):
        assert base_kwargs.pop('stride', 1) == 1
        super().__init__(**base_kwargs)
        self.conv = None

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        assert c_out - s_in.num_features() >= 0
        self.conv = nn.Conv2d(s_in.num_features(), c_out, kernel_size=1, stride=1, padding=0, bias=False)
        return self.probe_outputs(s_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def config(self, finalize=True, **_) -> dict:
        cfg = super().config(finalize=finalize, **_)
        if finalize:
            cfg['name'] = SkipLayer.__name__
        return cfg
