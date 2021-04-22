import torch.nn as nn
from uninas.training.initializers.abstract import AbstractInitializer
from uninas.modules.modules.abstract import AbstractModule
from uninas.utils.loggers.python import logging
from uninas.register import Register


@Register.initializer()
class SPOSInitializer(AbstractInitializer):
    """
    Initialize weights as in Single Path One Shot
    adapted from https://github.com/megvii-model/SinglePathOneShot/blob/master/src/Search/network.py#L94
    """

    def _initialize_weights(self, net: AbstractModule, logger: logging.Logger):
        for name, m in net.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'stem' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
