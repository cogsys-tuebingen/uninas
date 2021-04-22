import torch
from uninas.data.abstract import AbstractDataSet
from uninas.models.networks.abstract import AbstractNetwork
from uninas.training.metrics.abstract import AbstractLogMetric, ResultValue
from uninas.utils.args import Argument, Namespace
from uninas.utils.misc import split
from uninas.register import Register


@Register.metric(only_head=True)
class AccuracyMetric(AbstractLogMetric):
    """
    Measure the accuracy in a classification problem, using only the last model head
    """

    def get_log_name(self) -> str:
        return 'accuracy'

    @classmethod
    def from_args(cls, args: Namespace, index: int, data_set: AbstractDataSet, head_weights: list) -> 'AbstractLogMetric':
        """
        :param args: global arguments namespace
        :param index: index of this metric
        :param data_set: data set that is evaluated on
        :param head_weights: how each head is weighted
        """
        assert data_set.is_classification(), "Accuracy can only be evaluated on classification data"
        all_parsed = cls._all_parsed_arguments(args, index=index)
        topk = split(all_parsed.pop('topk'), int)
        return cls(head_weights=head_weights, topk=topk, **all_parsed)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('topk', default='1, 5', type=str, help='log top k accuracy values'),
            Argument('ignore_index', default=-999, type=int,
                     help='if a target has this class index, ignore it. '
                          'if the network predicts this index, choose the next most-likely prediction instead.'),
        ]

    def _str_dict(self) -> dict:
        dct = super()._str_dict()
        dct.update({
            'topk': self.topk,
        })
        return dct

    @classmethod
    def accuracy(cls, outputs: torch.Tensor, targets: torch.Tensor, top_k=(1,)):
        """
        Computes the precision @k for the specified values of k between the network outputs and the targets

        :param outputs:
        :param targets: size [batch] with class indices
        :param top_k: tuple of 'k' values
        :return:
        """
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs[0]
        max_k = max(top_k)
        batch_size = targets.shape[0]

        _, predictions = outputs.topk(max_k, 1, True, True)
        predictions = predictions.t()
        correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

    def _evaluate(self, net: AbstractNetwork, inputs: torch.Tensor,
                  logits: [torch.Tensor], targets: torch.Tensor) -> {str: ResultValue}:
        """

        :param net: evaluated network
        :param inputs: network inputs
        :param logits: network outputs
        :param targets: output targets
        :return: dictionary of string keys with corresponding results
        """
        logits, targets = self._batchify_tensors([logits[-1]], targets)

        targets = self._remove_onehot(targets)

        logits, targets = self._ignore_with_index(logits, targets, self.ignore_index, self.ignore_index)
        logits = logits[0]

        return {str(self.topk[i]): ResultValue(v, count=targets.shape[0])
                for i, v in enumerate(self.accuracy(logits, targets, top_k=self.topk))}
