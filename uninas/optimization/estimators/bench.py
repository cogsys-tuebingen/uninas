"""
estimators (metrics) to rank different networks (architecture subsets of a supernet)
using the mini-benchmark data
"""

from uninas.optimization.estimators.abstract import AbstractEstimator
from uninas.optimization.benchmarks.mini.benchmark import MiniNASBenchmark
from uninas.register import Register


class AbstractMiniBenchEstimator(AbstractEstimator):
    def __init__(self, *args_, mini_api: MiniNASBenchmark, data_set: str = None, **kwargs):
        super().__init__(*args_, **kwargs)
        self.mini_api = mini_api
        self.data_set = data_set

    def _evaluate_tuple(self, values: tuple, strategy_name: str = None) -> float:
        """
        NOTE: either this or the _evaluate_batch method must be implemented in subclasses
        evaluate a single tuple

        :param values: tuple
        :param strategy_name: None if candidate is global, otherwise specific to this weight strategy
        :return: single float value of how well the given parameter values do
        """
        assert strategy_name is None, "Can not check partial bench architectures"
        return self._evaluate_tuple_in_bench(values)

    def _evaluate_tuple_in_bench(self, values: tuple):
        raise NotImplementedError


@Register.hpo_estimator(requires_bench=True)
class MiniBenchLossEstimator(AbstractMiniBenchEstimator):
    """
    Checking the network loss in the mini-bench
    """

    def _evaluate_tuple_in_bench(self, values: tuple):
        return self.mini_api.get_by_arch_tuple(values).get_loss(self.data_set)


@Register.hpo_estimator(requires_bench=True)
class MiniBenchAcc1Estimator(AbstractMiniBenchEstimator):
    """
    Checking the network top-1 accuracy in the mini-bench
    """

    def _evaluate_tuple_in_bench(self, values: tuple):
        return self.mini_api.get_by_arch_tuple(values).get_acc1(self.data_set)


@Register.hpo_estimator(requires_bench=True)
class MiniBenchParamsEstimator(AbstractMiniBenchEstimator):
    """
    Checking the network parameter count in the mini-bench
    """

    def _evaluate_tuple_in_bench(self, values: tuple):
        return self.mini_api.get_by_arch_tuple(values).get_params(self.data_set)


@Register.hpo_estimator(requires_bench=True)
class MiniBenchFlopsEstimator(AbstractMiniBenchEstimator):
    """
    Checking the network FLOPs in the mini-bench
    """

    def _evaluate_tuple_in_bench(self, values: tuple):
        return self.mini_api.get_by_arch_tuple(values).get_flops(self.data_set)


@Register.hpo_estimator(requires_bench=True)
class MiniBenchLatencyEstimator(AbstractMiniBenchEstimator):
    """
    Checking the network latency in the mini-bench
    """

    def _evaluate_tuple_in_bench(self, values: tuple):
        return self.mini_api.get_by_arch_tuple(values).get_latency(self.data_set)
