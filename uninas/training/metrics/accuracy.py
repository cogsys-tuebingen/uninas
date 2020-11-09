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


def accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk=(1,)):
    """
    Computes the precision @k for the specified values of k

    :param outputs:
    :param targets: either size [batch] with class indices, or [batch, class weights] in which case the argmax is taken
    :param topk: tuple of 'k' values
    :return:
    """
    if not isinstance(outputs, torch.Tensor):
        outputs = outputs[0]
    if len(targets.shape) > 1:
        targets = torch.argmax(targets, dim=-1)
    maxk = max(topk)
    batch_size = targets.shape[0]

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


@Register.metric(only_head=True)
class AccuracyMetric(AbstractMetric):
    """
    Measure the accuracy in a classification problem, using only the last model head
    """

    def __init__(self, args: Namespace, index: int, weights: list):
        super().__init__(args, index, weights)
        self.topk = split(self._parsed_argument('topk', args, self.index), int)

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
        return {key+'/accuracy/'+str(self.topk[i]): v.unsqueeze(0)
                for i, v in enumerate(accuracy(logits[-1], targets, topk=self.topk))}

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('topk', default='1, 5', type=str, help='log top k accuracy values'),
        ]

    def _str_dict(self) -> dict:
        dct = super()._str_dict()
        dct.update({
            'topk': self.topk,
        })
        return dct
