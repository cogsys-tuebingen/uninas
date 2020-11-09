"""
trained models from the timm framework
https://github.com/rwightman/pytorch-image-models
"""


from typing import Union
import torch
import torch.nn as nn
from timm.models.factory import create_model
from uninas.networks.abstract2 import Abstract2Network
from uninas.utils.args import Argument, Namespace
from uninas.utils.shape import Shape, ShapeList
from uninas.utils.loggers.python import get_logger


logger = get_logger()


class AbstractTimmNetwork(Abstract2Network):

    def __init__(self, model_name: str, checkpoint_path: str):
        super().__init__(model_name, checkpoint_path)
        self.input_shape = None
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

    @classmethod
    def from_args(cls, args: Namespace, index=None):
        """
        :param args: global argparse namespace
        :param index: index for the arguments
        """
        all_parsed = cls._all_parsed_arguments(args, index=index)
        name = all_parsed.pop('model_name')
        return cls(name, **all_parsed)

    def _build2(self, s_in: Shape, s_out: Shape):
        self.input_shape = s_in
        self.net = create_model(self.name,
                                in_chans=s_in.num_features,
                                num_classes=s_out.num_features)
        self.on_network_built(s_in, s_out)

        log_str = '  {:<8}{:<45}-> {:<45}'
        logger.info('Built %s (%s):' % (self.net.__class__.__name__, self.name))
        logger.info(log_str.format('index', 'input shapes', 'output shapes'))
        for i, (s_in, s_out) in enumerate(zip(self.get_input_shapes(flatten=True), self.get_output_shapes(flatten=True))):
            logger.info(log_str.format(i, s_in.str(), s_out.str()))
        return s_out

    def on_network_built(self, s_in: Shape, s_out: Shape):
        pass

    def get_network(self) -> nn.Module:
        return self.net

    def get_stem(self) -> nn.Module:
        raise NotImplementedError

    def get_cells(self) -> nn.ModuleList():
        raise NotImplementedError

    def get_heads(self) -> nn.ModuleList():
        raise NotImplementedError

    def _get_input_shapes(self) -> ShapeList:
        if self.get_cached('all_input_shapes') is None:
            self.cached['all_input_shapes'] = self._get_input_shapes2()
        return self.get_cached('all_input_shapes')

    def _get_input_shapes2(self) -> ShapeList:
        raise NotImplementedError

    def _get_output_shapes(self) -> ShapeList:
        if self.get_cached('all_output_shapes') is None:
            self.cached['all_output_shapes'] = self._get_output_shapes2()
        return self.get_cached('all_output_shapes')

    def _get_output_shapes2(self) -> ShapeList:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, **kwargs) -> [torch.Tensor]:
        return [self.net(x, **kwargs)]

    def str(self, depth=0, **_) -> str:
        return ""

    def config(self, **_) -> Union[None, dict]:
        return None
