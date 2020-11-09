"""
estimators (metrics) to rank different networks (architecture subsets of a supernet)
using the mini NAS-Bench-201 data
"""

from uninas.optimization.common.estimators.abstract import AbstractEstimator
from uninas.benchmarks.mini import MiniNASBenchApi
from uninas.register import Register


class AbstractMiniBenchEstimator(AbstractEstimator):
    def __init__(self, *args_, mini_api: MiniNASBenchApi, mini_api_set='cifar10', **kwargs):
        super().__init__(*args_, **kwargs)
        self.mini_api = mini_api
        self.mini_api_set = mini_api_set

    def evaluate_tuple(self, values: tuple, strategy_name: str = None):
        """
        :param values: architecture description
        :param strategy_name: None if candidate is global, otherwise specific to this weight strategy
        """
        assert strategy_name is None, "Can not check partial bench architectures"
        return self._evaluate_tuple(values)

    def _evaluate_tuple(self, values: tuple):
        raise NotImplementedError


@Register.hpo_estimator()
class MiniBenchLossEstimator(AbstractMiniBenchEstimator):
    """
    Checking the network loss in the mini-bench
    """

    def _evaluate_tuple(self, values: tuple):
        return self.mini_api.get_by_arch_tuple(values).get_loss(self.mini_api_set)


@Register.hpo_estimator()
class MiniBenchAcc1Estimator(AbstractMiniBenchEstimator):
    """
    Checking the network top-1 accuracy in the mini-bench
    """

    def _evaluate_tuple(self, values: tuple):
        return self.mini_api.get_by_arch_tuple(values).get_acc1(self.mini_api_set)


@Register.hpo_estimator()
class MiniBenchParamsEstimator(AbstractMiniBenchEstimator):
    """
    Checking the network parameter count in the mini-bench
    """

    def _evaluate_tuple(self, values: tuple):
        return self.mini_api.get_by_arch_tuple(values).get_params(self.mini_api_set)


@Register.hpo_estimator()
class MiniBenchFlopsEstimator(AbstractMiniBenchEstimator):
    """
    Checking the network FLOPs in the mini-bench
    """

    def _evaluate_tuple(self, values: tuple):
        return self.mini_api.get_by_arch_tuple(values).get_flops(self.mini_api_set)


@Register.hpo_estimator()
class MiniBenchLatencyEstimator(AbstractMiniBenchEstimator):
    """
    Checking the network latency in the mini-bench
    """

    def _evaluate_tuple(self, values: tuple):
        return self.mini_api.get_by_arch_tuple(values).get_latency(self.mini_api_set)
