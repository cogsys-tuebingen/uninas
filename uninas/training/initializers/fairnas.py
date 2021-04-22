import math
import torch.nn as nn
from uninas.training.initializers.abstract import AbstractInitializer
from uninas.modules.modules.abstract import AbstractModule
from uninas.utils.loggers.python import logging
from uninas.register import Register


@Register.initializer()
class FairNasInitializer(AbstractInitializer):
    """
    Initialize weights as in FairNAS / Scarlet-NAS
    adapted from https://github.com/xiaomi-automl/FairNAS/blob/master/models/FairNAS_A.py#L118
    """

    def _initialize_weights(self, net: AbstractModule, logger: logging.Logger):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                if m.bias is not None:
                    m.bias.data.zero_()
