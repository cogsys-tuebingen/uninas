import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from uninas.data.abstract import AbstractDataSet
from uninas.models.networks.abstract import AbstractNetwork
from uninas.training.metrics.abstract import AbstractAccumulateMetric, ResultValue
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


try:
    from scipy.stats import norm, kstest


    @Register.metric(only_head=True)
    class FitDistributionsMetric(AbstractAccumulateMetric):
        """
        Measure the differences between prediction and targets,
        try to fit distributions to them,
        create a histogram
        """

        def get_log_name(self) -> str:
            return "distributions"

        def _str_dict(self) -> dict:
            dct = super()._str_dict()
            dct.update({
                'normal': self.fit_normal,
            })
            return dct

        @classmethod
        def _combine_tensors(cls, dict_key: str, tensors: [torch.Tensor]) -> torch.Tensor:
            """ how to combine tensors if they are gathered from distributed training or from different batches """
            if dict_key == "max_value":
                return max(tensors)
            return torch.cat(tensors, dim=0)

        @classmethod
        def from_args(cls, args: Namespace, index: int, data_set: AbstractDataSet, head_weights: list) -> AbstractAccumulateMetric:
            """
            :param args: global arguments namespace
            :param index: index of this metric
            :param data_set: data set that is evaluated on
            :param head_weights: how each head is weighted
            """
            assert data_set.is_regression(), "Can only compute difference metrics for regression data"
            assert data_set.get_label_shape().numel() == 1, "Can only compute difference metrics for single values"
            return super().from_args(args, index, data_set, head_weights)

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return super().args_to_add(index) + [
                Argument('num_bins', default=100, type=int, help='number of histogram bins'),
                Argument('log_y', default="False", type=str, help='plot the y axis in log scale', is_bool=True),
                Argument('fit_normal', default="False", type=str, help='fit an approximated normal distribution', is_bool=True),
                Argument('test_ks', default="False", type=str, help='test distributions with Kolmogorov-Smirnov', is_bool=True),
            ]

        def _evaluate(self, net: AbstractNetwork, inputs: torch.Tensor,
                      logits: [torch.Tensor], targets: torch.Tensor) -> {str: ResultValue}:
            """

            :param net: evaluated network
            :param inputs: network inputs
            :param logits: network outputs
            :param targets: output targets
            :return: dictionary of string keys with corresponding results
            """
            return {"diff": ResultValue((targets - logits[-1]).squeeze().cpu())}

        def _update_stats(self, stats: dict) -> dict:
            """
            pre-compute things on the stats that may be shared across log/viz

            :param stats: accumulated stats throughout the _evaluate calls
            :return: stats
            """
            values = stats["diff"].squeeze().numpy()
            stats = dict(values=values, fit={}, test={})
            # add density functions, each has fun(x)
            if self.fit_normal:
                mu, std = norm.fit(values)
                stats["fit"]["norm"] = (lambda x: norm.pdf(x, mu, std), dict(mu=mu, std=std))
            # add test statistics for each density function
            test_any = any([self.test_ks])
            for k, (fit_fun, fit_kwargs) in stats["fit"].items():
                if test_any:
                    stats["test"][k] = dict()
                if self.test_ks:
                    statistic, pvalue = kstest(values, fit_fun)
                    stats["test"][k]["ks"] = dict(statistic=statistic, pvalue=pvalue)
            return stats

        def _log_stats(self, stats: dict) -> dict:
            """
            compute this metric

            :param stats: accumulated stats throughout the _evaluate() calls, after calling _update_stats() on them
            :return: log dict
            """
            dct = {}
            for fit_name, (fit_fun, fit_kwargs) in stats["fit"].items():
                for k, v in fit_kwargs.items():
                    dct['%s/param/%s' % (fit_name, k)] = v
            for fit_name, fit_tests in stats["test"].items():
                for test_name, test_dict in fit_tests.items():
                    for k, v in test_dict.items():
                        dct['%s/test/%s/%s' % (fit_name, test_name, k)] = v
            return dct

        def _viz_stats(self, save_path: str, stats: dict):
            """
            visualize this metric

            :param save_path: where to save
            :param stats: accumulated stats throughout the _evaluate() calls, after calling _update_stats() on them
            :return:
            """
            values = stats["values"]
            plt.clf()
            has_pdf = len(stats["fit"]) > 0
            plt.hist(values, self.num_bins, label="difference", density=has_pdf, alpha=0.8)
            if has_pdf:
                x_min, x_max = plt.xlim()
                x = np.linspace(x_min, x_max, self.num_bins*5)
                for name, (fit_fun, fit_kwargs) in stats["fit"].items():
                    p = fit_fun(x)
                    kwargs_str = ", ".join(["%s=%.3f" % (k, v) for k, v in fit_kwargs.items()])
                    plt.plot(x, p, label="%s(%s)" % (name, kwargs_str))
            plt.xlabel("differences")
            plt.ylabel("density of %d values" % values.shape[0] if self.fit_normal else "num occurrences")
            plt.legend()
            if self.log_y:
                plt.yscale("log")
            if isinstance(save_path, str):
                path = "%s.pdf" % save_path
                os.makedirs(os.path.dirname(path), exist_ok=True)
                plt.savefig(path)

except ImportError as e:
    Register.missing_import(e)
