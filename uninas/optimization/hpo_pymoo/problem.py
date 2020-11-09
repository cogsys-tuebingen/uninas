import numpy as np

from pymoo.model.problem import Problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from uninas.optimization.common.estimators.abstract import AbstractEstimator
from uninas.benchmarks.mini import MiniNASBenchApi


class PymooProblem(Problem):
    """
    A problem, optimized by a pymoo algorithm
    """

    def __init__(self, estimators: [AbstractEstimator], strategy_name: str = None, **kwargs):
        self.estimators = sorted(estimators, key=lambda e: e.is_constraint(), reverse=True)
        self.strategy_name = strategy_name
        super().__init__(n_obj=sum([1 if e.is_objective() else 0 for e in self.estimators]),
                         n_constr=sum([1 if e.is_constraint() else 0 for e in self.estimators]),
                         elementwise_evaluation=True,
                         **kwargs)

    def objectives(self) -> [AbstractEstimator]:
        return [e for e in self.estimators if e.is_objective()]

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
        values, constraints, in_constraints = [], [], True
        for e in self.estimators:
            v, c = e.evaluate_pymoo(x, in_constraints, strategy_name=self.strategy_name)
            if v is not None:
                values.append(v)
            if c is not None:
                in_constraints = in_constraints and c < 0
                constraints.append(c)
        out["F"] = np.array(values)
        if len(constraints) > 0:
            out["G"] = np.array(constraints)

    def plottable_pareto_front(self) -> (np.array, np.array):
        return [], []


class BenchPymooProblem(PymooProblem):
    """
    A problem, optimized by a pymoo algorithm
    """

    def __init__(self, estimators: [AbstractEstimator], mini_api: MiniNASBenchApi, calc_pareto=False):
        xl, xu = mini_api.get_space_lower_upper()
        self.calc_pareto = calc_pareto
        super().__init__(n_var=len(xl), estimators=estimators, xl=xl, xu=xu)
        self.mini_api = mini_api
        self._valid_pareto_idx = None
        self._valid_results = dict()  # only those within constraints

    def iterate(self) -> list:
        """ iterate the entire discrete search space, returning lists """
        def rec(empty: list, depth=0):
            if depth >= self.n_var:
                yield empty
            else:
                for v in range(self.xl[depth], self.xu[depth]+1):
                    for lst in rec(empty, depth=depth+1):
                        h = lst.copy()
                        h[depth] = v
                        yield h
        return [x for x in rec([0 for _ in range(self.n_var)])]

    def _calc_pareto_ranks(self):
        # indices of the pareto front ranks
        if self._valid_pareto_idx is None and self.calc_pareto:
            # first eval all
            self._valid_results["F"], all_x = [], self.iterate()
            out = dict()
            for i, x in enumerate(all_x):
                self._evaluate(x, out)
                if out.get('G', -1) < 0:
                    self._valid_results["F"].append(out['F'])
            all_f = np.row_stack(self._valid_results["F"])
            nds = NonDominatedSorting(method='fast_non_dominated_sort')
            # find best in small subgroups
            group_size, best_idx = 200, []
            for i in range(0, len(self._valid_results["F"]) // group_size + 1):
                start, end = group_size*i, min(group_size*(i+1), len(self._valid_results["F"]))
                idx = np.arange(start, end)
                group = all_f[idx]
                pareto = nds.do(group, only_non_dominated_front=True) + start
                best_idx.append(pareto)
            # concat subgroups and find best
            best_idx = np.concatenate(best_idx, axis=0)
            pareto = nds.do(all_f[best_idx], only_non_dominated_front=True)
            self._valid_pareto_idx = best_idx[pareto]

    def _calc_pareto_front(self, flatten=True, **kwargs):
        # Pareto-front - not necessary but used for plotting
        if self.calc_pareto:
            self._calc_pareto_ranks()
            r = [self._valid_results["F"][i] for i in self._valid_pareto_idx]
            r = sorted(r, key=lambda k: k[0])
            return np.row_stack(r)

    def plottable_pareto_front(self) -> (np.array, np.array):
        front = self._calc_pareto_front()
        signs = self.objective_signs(only_minimize=True)
        return front[:, 0]*signs[0], front[:, 1]*signs[1]
