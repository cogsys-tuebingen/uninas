"""
trained models from the timm framework
https://github.com/rwightman/pytorch-image-models
"""


from typing import Union
import torch
import torch.nn as nn
from uninas.models.networks.abstract2 import AbstractExternalNetwork
from uninas.utils.args import Argument
from uninas.utils.shape import Shape, ShapeList
from uninas.register import Register


try:
    from timm.models.factory import create_model


    class AbstractTimmNetwork(AbstractExternalNetwork):

        def __init__(self, model_name: str, *args, **kwargs):
            super().__init__(model_name=model_name, *args, **kwargs)
            self.net = None

        @classmethod
        def _available_models(cls) -> [str]:
            raise NotImplementedError

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return super().args_to_add(index) + [
                Argument('model_name', default='', type=str, help='model name', choices=cls._available_models()),
            ]

        def _build2(self, s_in: Shape, s_out: Shape) -> ShapeList:
            """ build the network """
            self.net = create_model(self.model_name,
                                    in_chans=s_in.num_features(),
                                    num_classes=s_out.num_features())
            return self.get_network_output_shapes()

        def get_network(self) -> nn.Module:
            return self.net

        def get_stem(self) -> nn.Module:
            raise NotImplementedError

        def get_cells(self) -> nn.ModuleList():
            raise NotImplementedError

        def get_heads(self) -> nn.ModuleList():
            raise NotImplementedError

        def all_forward(self, x: torch.Tensor) -> [torch.Tensor]:
            """
            returns list of all heads' outputs
            the heads are sorted by ascending cell order
            """
            raise NotImplementedError

        def specific_forward(self, x: Union[torch.Tensor, list], start_cell=-1, end_cell=None) -> [torch.Tensor]:
            """
            can execute specific part of the network,
            returns result after end_cell
            """
            raise NotImplementedError

        def str(self, depth=0, **_) -> str:
            return ""

        def config(self, **_) -> Union[None, dict]:
            return None

except ImportError as e:
    Register.missing_import(e)
