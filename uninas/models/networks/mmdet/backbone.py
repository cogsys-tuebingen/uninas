from typing import Union
import torch
import torch.nn as nn
from uninas.register import Register


try:
    from mmcv import Config, DictAction
    from mmdet.models.builder import build_detector
    from mmdet.models.backbones.darknet import Darknet

    from uninas.models.networks import AbstractMMDetNetwork


    @Register.network(external=True)
    class BackboneMMDetNetwork(AbstractMMDetNetwork):
        """
        A network using the backbone of an MMDet network
        https://github.com/open-mmlab/mmdetection
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._heads = None

        def _build_heads(self):
            self._heads = nn.ModuleList([
                nn.Identity() for _ in range(self.get_num_backbone_outputs())
            ])

        def get_heads(self) -> nn.ModuleList():
            return self._heads

        def all_forward(self, x: torch.Tensor) -> [torch.Tensor]:
            """
            returns list of all heads' outputs
            the heads are sorted by ascending cell order
            """
            """
            # neck+head
            x = self.net.extract_feat(x)
            outs = self.net.bbox_head(x)
            return outs[0]
            """
            x = self.net.backbone(x)  # is a tuple
            return [h(xs) for xs, h in zip(x, self._heads)]

        def specific_forward(self, x: Union[torch.Tensor, list], start_cell=-1, end_cell=None) -> [torch.Tensor]:
            """
            can execute specific part of the network,
            returns result after end_cell
            """
            if isinstance(x, list):
                assert len(x) == 1
                x = x[0]

            # stem, -1
            if start_cell <= -1:
                x = self.get_stem()(x)
            if end_cell == -1:
                return [x]

            # backbone blocks, 0 to n
            for i, b in enumerate(self.get_cells()):
                if start_cell <= i:
                    x = b(x)
                if end_cell == i:
                    return [x]

            """
            # neck+head
            if self.with_neck:
                x = self.neck(x)
            return list(self.bbox_head(x))
            """
            return list(x)  # is a tuple


except ImportError as e:
    Register.missing_import(e)
