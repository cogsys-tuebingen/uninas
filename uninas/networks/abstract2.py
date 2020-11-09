"""
this only exists to solve a cyclic dependency between Checkpointer and AbstractNetwork
"""


import torch.nn as nn
from uninas.networks.abstract import AbstractNetwork
from uninas.training.callbacks.checkpoint import CheckpointCallback
from uninas.utils.args import Namespace
from uninas.utils.torch.misc import count_parameters
from uninas.utils.shape import Shape, ShapeList
from uninas.utils.paths import find_pretrained_weights_path
from uninas.utils.loggers.python import get_logger

logger = get_logger()


class Abstract2Network(AbstractNetwork):

    @classmethod
    def from_args(cls, args: Namespace, index=None):
        """
        :param args: global argparse namespace
        :param index: argument index
        """
        raise NotImplementedError

    def _get_input_shapes(self) -> ShapeList:
        """ input shape of each cell in order """
        raise NotImplementedError

    def _get_output_shapes(self) -> ShapeList:
        """ output shape of each cell in order """
        raise NotImplementedError

    def get_network(self) -> nn.Module:
        raise NotImplementedError

    def get_stem(self) -> nn.Module:
        raise NotImplementedError

    def get_cells(self) -> nn.ModuleList():
        raise NotImplementedError

    def get_heads(self) -> nn.ModuleList():
        raise NotImplementedError

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        s_out = self._build2(s_in, s_out)
        logger.info('Network built, it has %d parameters!' % count_parameters(self))
        if len(self.checkpoint_path) > 0:
            path = find_pretrained_weights_path(self.checkpoint_path, self.name,
                                                raise_missing=len(self.checkpoint_path) > 0)
            self.loaded_weights(CheckpointCallback.load_network(path, self.get_network()))
        return s_out

    def _build2(self, s_in: Shape, s_out: Shape) -> Shape:
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError
