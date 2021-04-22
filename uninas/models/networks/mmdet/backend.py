"""
utility for the MMDet network handling
as the backbones/necks etc have no common interface for some required utility (e.g. get cells), define it here
"""


import sys
import inspect
import torch.nn as nn
from uninas.register import Register


try:
    from mmcv import Config, DictAction
    from mmdet.models.builder import build_detector
    from mmdet.models.backbones.darknet import Darknet


    class AbstractBackboneInterface:
        accepted_cls = []

        def __init__(self, backbone: nn.Module):
            assert self.is_accepted_class(backbone), "Given class (%s) not in list of accepted classes of (%s)"\
                                                     % (backbone.__class__.__name__, self.__class__.__name__)
            self.backbone = backbone
            self._cells = None
            self._stem = None

        @classmethod
        def is_accepted_class(cls, backbone: nn.Module):
            return any([isinstance(backbone, ac) for ac in cls.accepted_cls])

        def get_stem(self) -> nn.Module:
            raise NotImplementedError

        def get_num_outputs(self) -> int:
            raise NotImplementedError

        def get_cells(self) -> nn.ModuleList():
            if self._cells is None:
                self._cells = self._get_cells()
            return self._cells

        def _get_cells(self) -> nn.ModuleList():
            raise NotImplementedError


    class DarknetBackboneInterface(AbstractBackboneInterface):
        accepted_cls = [Darknet]

        def get_stem(self) -> nn.Module:
            return self.backbone.conv1

        def get_num_outputs(self) -> int:
            return len(self.backbone.out_indices)

        def _get_cells(self) -> nn.ModuleList():
            cells = []
            for name in self.backbone.cr_blocks[1:]:
                cells.append(getattr(self.backbone, name))
            return nn.ModuleList(cells)


    def get_backbone_interface(backbone: nn.Module) -> AbstractBackboneInterface:
        for _, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
            if issubclass(cls, AbstractBackboneInterface) and cls.is_accepted_class(backbone):
                return cls(backbone)
        raise NotImplementedError("Backbone %s can not be handled yet" % backbone.__class__.__name__)


except ImportError as e:
    Register.missing_import(e)
