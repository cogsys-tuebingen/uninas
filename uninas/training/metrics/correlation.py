import os
import matplotlib.pyplot as plt
import torch
from uninas.data.abstract import AbstractDataSet
from uninas.models.networks.abstract import AbstractNetwork
from uninas.training.metrics.abstract import AbstractAccumulateMetric, ResultValue
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


@Register.metric(only_head=True)
class CorrelationsMetric(AbstractAccumulateMetric):
    """
    Measure any NAS correlation as a metric
    """

    def get_log_name(self) -> str:
        return 'correlation'

    @classmethod
    def _combine_tensors(cls, dict_key: str, tensors: [torch.Tensor]) -> torch.Tensor:
        """ how to combine tensors if they are gathered from distributed training or from different batches """
        return torch.cat(tensors, dim=0)

    @classmethod
    def from_args(cls, args: Namespace, index: int, data_set: AbstractDataSet, head_weights: list) -> 'AbstractAccumulateMetric':
        """
        :param args: global arguments namespace
        :param index: index of this metric
        :param data_set: data set that is evaluated on
        :param head_weights: how each head is weighted
        """
        assert data_set.is_regression(), "Can only compute correlation metrics for regression data"
        assert data_set.get_label_shape().numel() == 1, "Can only compute correlation metrics for single values"
        all_parsed = cls._all_parsed_arguments(args, index=index)
        correlation_cls = cls._parsed_argument('correlations', args, index, split_=str)
        correlation_cls = [Register.nas_metrics.get(c) for c in correlation_cls]
        return cls(head_weights=head_weights, correlation_cls=correlation_cls, **all_parsed)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('correlations', default='KendallTauNasMetric, SpearmanNasMetric', type=str,
                     help='NAS correlation metrics to evaluate, only possible if the classes have no arguments themselves'),
        ]

    def _str_dict(self) -> dict:
        dct = super()._str_dict()
        dct.update({
            'correlations': '[%s]' % ', '.join([c.short_name() for c in self.correlation_cls]),
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
        return {
            "logits": ResultValue(logits[-1]),
            "targets": ResultValue(targets),
        }

    def _compute_stats(self, save_dir: str, key: str, stats: dict) -> dict:
        logits, targets = stats["logits"].cpu().numpy(), stats["targets"].cpu().numpy()
        return {c.short_name(): ResultValue(c.correlation_value(logits, targets), count=targets.shape[0])
                for c in self.correlation_cls}

    def _viz_stats(self, save_dir: str, key: str, prefix: str, stats: dict):
        """ visualize this metric """
        logits, targets = stats["logits"].cpu().numpy(), stats["targets"].cpu().numpy()
        label = ", ".join(["%s=%.2f" % (c.short_name(), c.correlation_value(logits, targets)) for c in self.correlation_cls])
        x0, x1 = min(targets), max(targets)
        plt.clf()
        plt.plot((x0, x1), (x0, x1), color="red")
        plt.scatter(logits, targets, label=label, color="blue", s=0.5)
        plt.legend()
        plt.xlabel("predictions")
        plt.ylabel("targets")
        if isinstance(save_dir, str):
            path = "%s/%s/%s.pdf" % (save_dir, prefix, key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
