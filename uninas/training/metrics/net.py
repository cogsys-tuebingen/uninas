"""
metrics to measure an individual network's performance during training
to be used by lightning models to extend tqdm dictionaries
"""


import torch
from uninas.networks.abstract import AbstractNetwork
from uninas.training.metrics.abstract import AbstractMetric
from uninas.utils.args import Argument, Namespace
from uninas.utils.misc import split
from uninas.register import Register


@Register.metric()
class MacsMetric(AbstractMetric):
    """
    Measure the macs
    """

    def evaluate(self, net: AbstractNetwork,
                 inputs: torch.Tensor, logits: [torch.Tensor], targets: torch.Tensor,
                 key: str) -> {str: [torch.Tensor]}:
        """

        :param net: evaluated network
        :param inputs: network inputs
        :param logits: network outputs
        :param targets: output targets
        :param key: prefix for the dict keys, e.g. "train" or "test"
        :return: dictionary of string keys with corresponding scalar tensors
        """
        return {key+'/net/macs': net.profile_macs(inputs)}
