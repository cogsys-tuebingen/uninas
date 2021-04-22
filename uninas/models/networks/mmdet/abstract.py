"""
trained models from the mmdet framework
https://github.com/open-mmlab/mmdetection
"""


from typing import Union
import torch
import torch.nn as nn
from uninas.models.networks.abstract2 import AbstractExternalNetwork
from uninas.utils.args import Argument
from uninas.utils.shape import Shape, ShapeList
from uninas.utils.paths import maybe_download, FileType
from uninas.register import Register


try:
    from mmcv import Config, DictAction
    from mmdet.models.builder import build_detector

    from uninas.models.networks import get_backbone_interface


    class AbstractMMDetNetwork(AbstractExternalNetwork):

        def __init__(self, config_path: str, use_config_pretrained: bool, *args, **kwargs):
            # mmdet config
            config_path = maybe_download(config_path, FileType.MISC)
            cfg = Config.fromfile(config_path)
            try:
                self._mmdet_model = cfg.model
            except:
                self._mmdet_model = cfg

            # super
            super().__init__(model_name=self._mmdet_model.type, *args, **kwargs)
            self.input_shape = None
            self.net = None
            self._backbone_interface = None

            # handle pretrained
            if use_config_pretrained:
                if isinstance(self._mmdet_model.pretrained, str) and len(self._mmdet_model.pretrained) > 0:
                    self.loaded_weights(True)
                self.set(checkpoint_path="")
            else:
                self._mmdet_model.pretrained = None

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return super().args_to_add(index) + [
                Argument('config_path', default='', type=str, help='path to the mmdet config', is_path=True),
                Argument('use_config_pretrained', default='False', type=str, is_bool=True,
                         help='Use the pretrained weights defined in the config'),
            ]

        def _build2(self, s_in: Shape, s_out: Shape) -> ShapeList:
            """ build the network """
            self.input_shape = s_in
            self.net = build_detector(self._mmdet_model)
            self._backbone_interface = get_backbone_interface(self.net.backbone)
            self._build_heads()
            return self.get_network_output_shapes()

        def _build_heads(self):
            pass

        def get_network(self) -> nn.Module:
            return self.net

        def get_num_backbone_outputs(self) -> int:
            return self._backbone_interface.get_num_outputs()

        def get_stem(self) -> nn.Module:
            return self._backbone_interface.get_stem()

        def get_cells(self) -> nn.ModuleList():
            return self._backbone_interface.get_cells()

        def get_heads(self) -> nn.ModuleList():
            raise NotImplementedError

        def _get_cell_input_shapes_uncached(self) -> ShapeList:
            shapes = ShapeList([self.input_shape])
            shapes.extend(self._get_cell_output_shapes())
            return shapes[:-1]

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


except ImportError as e:
    Register.missing_import(e)
