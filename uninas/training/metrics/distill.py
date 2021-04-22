import torch
import torch.nn.functional as F
from uninas.models.networks.abstract import AbstractNetwork
from uninas.training.metrics.abstract import AbstractLogMetric
from uninas.training.result import ResultValue
from uninas.register import Register


@Register.metric(distill=True)
class DistillL2Metric(AbstractLogMetric):
    """
    Measure the distillation L2 loss between tensors
    which are the intermediate network feature outputs, of student and teacher networks
    """

    def get_log_name(self) -> str:
        return 'L2'

    def _evaluate(self, net: AbstractNetwork, inputs: torch.Tensor,
                  outputs: [[(torch.Tensor, torch.Tensor)]], targets: torch.Tensor)\
            -> {str: ResultValue}:
        """

        :param net: evaluated network
        :param inputs: network inputs
        :param outputs: network outputs
        :param targets: output targets
        :return: dictionary of string keys with corresponding results
        """
        dct = {}
        for i1, (w, o1) in enumerate(zip(self.head_weights, outputs)):
            for i2, (o21, o22) in enumerate(o1):
                dct['%d/%d' % (i1, i2)] = ResultValue(F.mse_loss(o21, o22) * w, o21.shape[0])
        return dct
