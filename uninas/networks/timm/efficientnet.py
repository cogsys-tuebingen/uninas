import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.efficientnet import default_cfgs
from uninas.networks.timm.abstract import AbstractTimmNetwork
from uninas.utils.shape import Shape, ShapeList
from uninas.register import Register


@Register.network(external=True)
class EfficientNetTimmNetwork(AbstractTimmNetwork):
    """
    An EfficientNet-based model from the pytorch-image-models framework
    """

    @classmethod
    def _available_models(cls) -> [str]:
        return list(default_cfgs.keys())

    def on_network_built(self, s_in: Shape, s_out: Shape):
        # replace the forward pass function
        def forward(self_, x: torch.Tensor, start_block=-1, end_block=None) -> torch.Tensor:
            # stem, -1
            if start_block <= -1:
                x = self_.conv_stem(x)
                x = self_.bn1(x)
                x = self_.act1(x)
            if end_block == -1:
                return x

            # blocks, 0 to n
            for i, b in enumerate(self_.blocks):
                if start_block <= i:
                    x = b(x)
                if end_block == i:
                    return x

            # head, otherwise
            x = self_.conv_head(x)
            x = self_.bn2(x)
            x = self_.act2(x)
            x = self_.global_pool(x)
            if self_.drop_rate > 0.:
                x = F.dropout(x, p=self_.drop_rate, training=self_.training)
            return self_.classifier(x)

        self.net.forward = types.MethodType(forward, self.net)

    def _set_dropout_rate(self, p=None):
        """ set the dropout rate of every dropout layer to p """
        self.net.drop_rate = p

    def get_stem(self) -> nn.Module:
        return nn.Sequential(self.net.conv_stem, self.net.bn1, self.net.act1)

    def get_cells(self) -> nn.ModuleList():
        return self.net.blocks

    def get_heads(self) -> nn.ModuleList():
        return nn.ModuleList([
            nn.Sequential(self.net.conv_head, self.net.bn2, self.net.act2, self.net.global_pool, self.net.classifier)
        ])

    def _get_input_shapes2(self) -> ShapeList:
        shapes = ShapeList([self.input_shape])
        shapes.extend(self._get_output_shapes())
        return shapes[:-1]

    def _get_output_shapes2(self) -> ShapeList:
        x = self.input_shape.random_tensor(batch_size=2)
        training = self.training
        self.train(False)

        shapes = ShapeList([])
        for i in range(self.num_cells()):
            y = self.net(x, start_block=-1, end_block=i)
            shapes.append(Shape.from_tensor(y))
        shapes.append(Shape.from_tensor(self.net(x)))

        self.train(training)
        return shapes
