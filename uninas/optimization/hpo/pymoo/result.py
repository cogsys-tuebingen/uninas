import numpy as np
import matplotlib.pyplot as plt
from pymoo.performance_indicator.hv import Hypervolume
from pymoo.model.result import Result
from pymoo.optimize import minimize
from uninas.utils.loggers.python import Logger, log_in_columns


class SingleResult:
    def __init__(self, x: np.array, f: np.array, g: np.array, cv: np.array):
        self.x = x
        self.f = f
        self.g = g
        self.cv = cv


class PymooResultWrapper:
    """
    Convenience wrapper for the pymoo result
    """

    def __init__(self, result: Result):
        self.result = result

    @classmethod
    def minimize(cls, *args, **kwargs):
        result = minimize(*args, **kwargs)
        return cls(result)

    def sorted_best(self, reverse=False) -> [SingleResult]:
        assert (self.result.X is not None) and (self.result.F is not None), "No valid parameters / results found"
        best = [SingleResult(x, f, g, cv)
                for x, f, g, cv in zip(self.result.X, self.result.F, self.result.G, self.result.CV)]
        return sorted(best, reverse=reverse, key=lambda sr: sr.f[0])

    def n_eval_by_iteration(self) -> list:
        return [a.evaluator.n_eval for a in self.result.history]

    def population_by_iteration(self) -> list:
        return [a.pop for a in self.result.history]

    def feasible_values_by_iteration(self) -> list:
        pops = self.population_by_iteration()
        return [p[p.get("feasible")[:, 0]].get("F") for p in pops]

    def feasible_population_by_iteration(self) -> list:
        pops = self.population_by_iteration()
        return [p[p.get("feasible")[:, 0]] for p in pops]

    def log_best(self, logger: Logger):
        signs = self.result.problem.objective_signs(only_minimize=True)
        rows = [['', 'gene'] + self.result.problem.objective_labels(only_minimize=False)]
        logger.info("best candidates:")
        for i, sr in enumerate(self.sorted_best()):
            rows.append([i, sr.x] + [v for v in sr.f*signs])
        log_in_columns(logger, rows, add_bullets=True, num_headers=1)

    def plot_all_f(self, checkpoint_dir: str, name='fa'):
        """ plot all populations over time """
        plt.clf()
        ax = plt.gca()
        cmap = plt.get_cmap('plasma')
        population_by_iteration = self.feasible_population_by_iteration()
        labels = self.result.problem.objective_labels(only_minimize=False)
        signs = self.result.problem.objective_signs(only_minimize=True)

        for i, population in [e for e in enumerate(population_by_iteration)]:
            x, y = [], []
            for ind in population:
                x.append(ind.F[0] * signs[0])
                y.append(ind.F[1] * signs[1])
            ax.scatter(x, y, label='__no_legend__', s=16,
                       c=[i]*len(x), cmap=cmap, vmin=0, vmax=len(population_by_iteration)-1)
        plt.plot(*self.result.problem.plottable_pareto_front(), color="black", alpha=0.7)

        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        try:
            col_bar = plt.colorbar(ax.get_children()[2], ax=ax)
            col_bar.set_label('iterations')
        except:
            pass
        plt.savefig('%s/%s.pdf' % (checkpoint_dir, name))

    def plot_hv(self, checkpoint_dir: str, name='hv'):
        """ plot the hyper-volume over time """
        metric = Hypervolume(ref_point=self.result.problem.get_ref_point())
        x = self.n_eval_by_iteration()
        hv = [metric.calc(f) for f in self.feasible_values_by_iteration()]
        plt.clf()
        plt.plot(x, hv, '-o')
        plt.xlabel("Function Evaluations")
        plt.ylabel("Hyper-Volume")
        plt.savefig('%s/%s.pdf' % (checkpoint_dir, name))
