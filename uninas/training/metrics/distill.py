"""
metrics to measure an individual network's performance during training
to be used by lightning models to extend tqdm dictionaries
"""


import torch
import torch.nn.functional as F
from uninas.networks.abstract import AbstractNetwork
from uninas.training.metrics.abstract import AbstractMetric
from uninas.register import Register


@Register.metric(distill=True)
class DistillL2Metric(AbstractMetric):
    """
    Measure the distillation L2 loss for each stage
    """

    def evaluate(self, net: AbstractNetwork,
                 inputs: torch.Tensor, outputs: [[(torch.Tensor, torch.Tensor)]], targets: torch.Tensor,
                 key: str) -> {str: [torch.Tensor]}:
        """

        :param net: evaluated network
        :param inputs: network inputs
        :param outputs: network outputs
        :param targets: output targets
        :param key: prefix for the dict keys, e.g. "train" or "test"
        :return: dictionary of string keys with corresponding scalar tensors
        """
        dct = {}
        for i1, (w, o1) in enumerate(zip(self.weights, outputs)):
            for i2, (o21, o22) in enumerate(o1):
                dct[key+'/L2/%d/%d' % (i1, i2)] = [F.mse_loss(o21, o22) * w]
        return dct
