"""
common interface to internal and external networks
"""


import torchprofile
import numpy as np
import torch.nn as nn
from uninas.model.modules.abstract import AbstractArgsModule
from uninas.utils.args import Namespace, Argument
from uninas.utils.shape import Shape, ShapeList
from uninas.utils.loggers.python import get_logger

logger = get_logger()


class AbstractNetwork(AbstractArgsModule):
    def __init__(self, name: str, checkpoint_path: str):
        super().__init__()
        self.name = name
        self.checkpoint_path = checkpoint_path
        self._loaded_weights = False

    @classmethod
    def from_args(cls, args: Namespace, index=None):
        """
        :param args: global argparse namespace
        :param index: argument index
        """
        raise NotImplementedError

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('checkpoint_path', default='', type=str, is_path=True,
                     help='use pretrained weights within the given local directory (matching by network name) or from an url'),
        ]

    def loaded_weights(self, b=True):
        self._loaded_weights = b

    def has_loaded_weights(self) -> bool:
        return self._loaded_weights

    def save_to_state_dict(self) -> dict:
        """ store additional info in the state dict """
        return dict()

    def load_from_state_dict(self, state: dict):
        """ load the stored additional info from the state dict """
        pass

    def get_head_weightings(self) -> [float]:
        """ get the weights of all heads, in order """
        return [1.0]*len(self.get_heads())

    def get_input_shapes(self, flatten=False) -> ShapeList:
        """ output shape of each cell in order """
        shapes = self._get_input_shapes()
        return shapes.flatten(flatten)

    def _get_input_shapes(self) -> ShapeList:
        """ input shape of each cell in order """
        raise NotImplementedError

    def get_output_shapes(self, flatten=False) -> ShapeList:
        """ output shape of each cell in order """
        shapes = self._get_output_shapes()
        return shapes.flatten(flatten)

    def _get_output_shapes(self) -> ShapeList:
        """ output shape of each cell in order """
        raise NotImplementedError

    def set_dropout_rate(self, p=None):
        """ set the dropout rate of every dropout layer to p """
        if isinstance(p, float):
            self._set_dropout_rate(p)

    def _set_dropout_rate(self, p: float):
        """ set the dropout rate of every dropout layer to p, no change for p=None """
        # set any dropout layer to p
        n = 0
        for m in self.get_network().modules():
            if isinstance(m, nn.Dropout):
                m.p = p
                n += 1
        assert n > 0 or p <= 0, "Could not set the dropout rate to %f, no nn.Dropout modules found!" % p

    def get_network(self) -> nn.Module:
        raise NotImplementedError

    def get_stem(self) -> nn.Module:
        raise NotImplementedError

    def num_cells(self) -> int:
        return len(self.get_cells())

    def get_cells(self) -> nn.ModuleList():
        raise NotImplementedError

    def get_heads(self) -> nn.ModuleList():
        raise NotImplementedError

    def _build2(self, s_in: Shape, s_out: Shape) -> Shape:
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def profile_macs(self, *inputs) -> np.int64:
        """
        measure the required macs (memory access costs) of a forward pass
        prevent randomly changing the architecture
        """
        return torchprofile.profile_macs(self.get_network(), args=inputs)
