"""
metrics to measure an individual network's performance during training
to be used by lightning models to extend tqdm dictionaries
"""


import torch
from uninas.networks.abstract import AbstractNetwork
from uninas.utils.args import ArgsInterface, Namespace


class AbstractMetric(ArgsInterface):
    def __init__(self, _: Namespace, index: int, weights: list):
        super().__init__()
        self.index = index
        self.weights = weights

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
        raise NotImplementedError
