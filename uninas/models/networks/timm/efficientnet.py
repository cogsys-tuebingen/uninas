from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from uninas.register import Register


try:
    from uninas.models.networks.timm.abstract import AbstractTimmNetwork
    from timm.models.efficientnet import default_cfgs


    @Register.network(external=True)
    class EfficientNetTimmNetwork(AbstractTimmNetwork):
        """
        An EfficientNet-based model from the pytorch-image-models framework
        """

        @classmethod
        def _available_models(cls) -> [str]:
            return list(default_cfgs.keys())

        def _set_dropout_rate(self, p=None) -> int:
            """ set the dropout rate of every dropout layer to p """
            self.net.drop_rate = p
            return 1

        def get_stem(self) -> nn.Module:
            return nn.Sequential(self.net.conv_stem, self.net.bn1, self.net.act1)

        def get_cells(self) -> nn.ModuleList():
            return self.net.blocks

        def get_heads(self) -> nn.ModuleList():
            return nn.ModuleList([
                nn.Sequential(self.net.conv_head, self.net.bn2, self.net.act2, self.net.global_pool, self.net.classifier)
            ])

        def all_forward(self, x: torch.Tensor) -> [torch.Tensor]:
            """
            returns list of all heads' outputs
            the heads are sorted by ascending cell order
            """
            return [self.net(x)]

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
                x = self.net.conv_stem(x)
                x = self.net.bn1(x)
                x = self.net.act1(x)
            if end_cell == -1:
                return [x]

            # blocks, 0 to n
            for i, b in enumerate(self.net.blocks):
                if start_cell <= i:
                    x = b(x)
                if end_cell == i:
                    return [x]

            # head, otherwise
            x = self.net.conv_head(x)
            x = self.net.bn2(x)
            x = self.net.act2(x)
            x = self.net.global_pool(x)
            if self.net.drop_rate > 0.:
                x = F.dropout(x, p=self.net.drop_rate, training=self.net.training)
            return [self.net.classifier(x)]


except ImportError as e:
    Register.missing_import(e)
