import numpy as np
from pymoo.model.problem import Problem
from uninas.optimization.estimators.abstract import AbstractEstimator
from uninas.optimization.benchmarks.mini.benchmark import MiniNASBenchmark


class PymooProblem(Problem):
    """
    A problem, optimized by a pymoo algorithm
    """

    def __init__(self, estimators: [AbstractEstimator], strategy_name: str = None, **kwargs):
        self.estimators = sorted(estimators, key=lambda e: e.is_constraint(), reverse=True)
        self.strategy_name = strategy_name
        super().__init__(n_obj=sum([1 if e.is_objective() else 0 for e in self.estimators]),
                         n_constr=sum([1 if e.is_constraint() else 0 for e in self.estimators]),
                         elementwise_evaluation=False,
                         **kwargs)

    def get_estimators(self) -> [AbstractEstimator]:
        return self.estimators

    def objectives(self) -> [AbstractEstimator]:
        return [e for e in self.get_estimators() if e.is_objective()]

    def objective_labels(self, only_minimize=True) -> [str]:
        return [e.name(only_minimize) for e in self.objectives()]

    def objective_signs(self, only_minimize=True) -> np.array:
        return np.array([e.sign(only_minimize) for e in self.objectives()])

    def get_ref_point(self, default: np.array = None) -> np.array:
        """ for hyper-volume """
        if default is None:
            default = np.array([0]*len(self.objectives()))
        return np.array([e.get_ref_point(default=d) for e, d in zip(self.objectives(), default)])

    def _evaluate(self, x: np.array, out: dict, *args, **kwargs):
        """
        evaluate a batch of parameters on the problem

        :param x: batch of parameters to evaluate
        :param out: dict to write to
                    ['F'] function evaluations
                    ['G'] constraints, violated where >0
        :param args:
        :param kwargs:
        """
        results, constraints, in_constraints = [], [], np.ones(shape=(x.shape[0],), dtype=np.bool)
        for e in self.get_estimators():
            r, c = e.evaluate_pymoo(x, in_constraints=in_constraints, strategy_name=self.strategy_name)
            results.append(r)
            if c is not None:
                in_constraints &= c
                constraints.append(c)
        out["F"] = np.column_stack(results)
        if len(constraints) > 0:
            # set -1 where everything is fine and 1 where the constraint is violated
            out["G"] = -(np.column_stack(constraints).astype(np.float32) - 0.5) * 2

    def plottable_pareto_front(self) -> (np.array, np.array):
        return [], []


class BenchPymooProblem(PymooProblem):
    """
    A problem, optimized by a pymoo algorithm
    """

    def __init__(self, estimators: [AbstractEstimator], mini_api: MiniNASBenchmark, calc_pareto=False):
        xl, xu = mini_api.get_my_space_lower_upper()
        self.calc_pareto = calc_pareto
        super().__init__(n_var=len(xl), estimators=estimators, xl=xl, xu=xu)
        self.mini_api = mini_api

    def plottable_pareto_front(self) -> (np.array, np.array):
        obj = self.objectives()
        keys = [o.key for o in obj]
        maximize = [o.is_maximize() for o in obj]
        entries = self.mini_api.get_all_sorted(sorted_by=keys, maximize=maximize, only_best=True)
        all_values = np.zeros(shape=(len(entries), len(keys)))
        for i, entry in enumerate(entries):
            all_values[i] = [entry.get(key) for key in keys]
        return all_values[:, 0], all_values[:, 1]
