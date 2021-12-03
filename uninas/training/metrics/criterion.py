import torch
from uninas.data.abstract import AbstractDataSet
from uninas.models.networks.abstract import AbstractNetwork
from uninas.training.criteria.abstract import AbstractCriterion, MultiCriterion
from uninas.training.metrics.abstract import AbstractLogMetric, ResultValue
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


@Register.metric(only_head=True)
class CriterionMetric(AbstractLogMetric):
    """
    Measure any criterion as a metric, not only as a loss function
    """

    def get_log_name(self) -> str:
        return 'criterion'

    @classmethod
    def from_args(cls, args: Namespace, index: int, data_set: AbstractDataSet, head_weights: list) -> 'CriterionMetric':
        """
        :param args: global arguments namespace
        :param index: index of this metric
        :param data_set: data set that is evaluated on
        :param head_weights: how each head is weighted
        """
        all_parsed = cls._all_parsed_arguments(args, index=index)
        criterion_str = all_parsed.pop('criterion')
        criterion_cls = Register.criteria.get(criterion_str)
        assert issubclass(criterion_cls, AbstractCriterion)
        criterion = criterion_cls(data_set)
        criterion = MultiCriterion(criterion, head_weights)
        return cls(head_weights=head_weights, criterion_str=criterion_str, criterion=criterion, **all_parsed)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('criterion', default='L1Criterion', type=str, choices=Register.criteria.names(),
                     help='criterion to evaluate (using its default arguments)'),
        ]

    def _str_dict(self) -> dict:
        dct = super()._str_dict()
        dct.update({
            'criterion': self.criterion_str,
        })
        return dct

    def _evaluate(self, net: AbstractNetwork, inputs: torch.Tensor,
                  logits: [torch.Tensor], targets: torch.Tensor) -> {str: ResultValue}:
        """

        :param net: evaluated network
        :param inputs: network inputs
        :param logits: network outputs
        :param targets: output targets
        :return: dictionary of string keys with corresponding results
        """
        v = self.criterion(logits, targets)
        return {self.criterion_str: ResultValue(v, count=targets.shape[0])}
