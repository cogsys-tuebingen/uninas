import random
import matplotlib.pyplot as plt
from collections import Iterable
from uninas.optimization.hpo.uninas.candidate import Candidate
from uninas.optimization.estimators.abstract import AbstractEstimator
from uninas.optimization.hpo.uninas.values import ValueSpace
from uninas.utils.misc import power_list


class Population:
    def __init__(self, name: str):
        self.name = name
        self.candidates = []
        self._fronts = []

    def __iter__(self) -> Iterable:
        for c in self.candidates:
            yield c

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def fronts(self) -> list:
        if len(self._fronts) > 0:
            return self._fronts
        return [self.candidates]

    @property
    def size(self) -> int:
        return len(self)

    def get_candidates(self) -> [Candidate]:
        return self.candidates

    def append(self, candidate: Candidate):
        self.candidates.append(candidate)

    def extend(self, candidates: [Candidate]):
        self.candidates.extend(candidates)

    def reduce_to_random_subset(self, n=1):
        """ keep only a random subset of all candidates """
        self.candidates = random.sample(self.candidates, min(n, len(self.candidates)))
        self._fronts = []

    def random_subset(self, n=1) -> [Candidate]:
        """ pick a random subset of all candidates """
        return random.sample(self.candidates, min(n, len(self.candidates)))

    def random_tourney_sample(self, num_participants=10) -> Candidate:
        """ pick random candidates and return the best """
        participants = self.random_subset(num_participants)
        best = participants[0]
        for participant in participants[1:]:
            if best.cur_ranked < participant.cur_ranked:
                continue
            if best.cur_crowding_dist < participant.cur_crowding_dist:
                continue
            best = participant
        return best

    def evaluate(self, estimators: [AbstractEstimator], strategy_name: str = None):
        """ evaluate all candidates in the population """
        for candidate in self.candidates:
            for estimator in estimators:
                estimator.evaluate_candidate(candidate, strategy_name=strategy_name)

    @staticmethod
    def is_dominated(candidate1: Candidate, candidate2: Candidate, objectives: [AbstractEstimator]) -> bool:
        """ check whether candidate1 is dominated by candidate2 """
        equal = []
        for objective in objectives:
            is_d, is_e = objective.is_dominated(candidate1, candidate2)
            if not is_d:
                return False
            equal.append(is_e)
        if all(equal):
            return False
        return True

    def sort_into_fronts(self, objectives: [AbstractEstimator]):
        """ all candidates are sorted into different ranks of pareto fronts, in which no individual dominates others """
        # figure out which candidates dominate which
        for c1 in self.candidates:
            c1.reset()
            for c2 in self.candidates:
                if self.is_dominated(c2, c1, objectives):
                    c1.cur_dominating.append(c2)
        # rank them
        for c in self.candidates:
            c.apply_rank()
        max_rank = max([c.cur_ranked for c in self.candidates])
        # add them to pareto fronts of different ranks, 0 being best
        self._fronts = [[] for _ in range(max_rank+1)]
        for c in self.candidates:
            self._fronts[c.cur_ranked].append(c)

    def sort_partially_into_fronts(self, objectives: [AbstractEstimator], num_dominated=0):
        """ all candidates are sorted into front 0 or 1, depending if they are dominated <= 'num_dominated' times """
        self._fronts = [[] for _ in range(2)]
        for c1 in self.candidates:
            c1.reset()
            n = 0
            for c2 in self.candidates:
                if self.is_dominated(c1, c2, objectives):
                    n += 1
                    if n > num_dominated:
                        break
            c1.apply_rank(0 if n <= num_dominated else 1)
            self._fronts[c1.cur_ranked].append(c1)

    @staticmethod
    def sort_within_front(front: [Candidate], objectives: [AbstractEstimator]):
        """ sort candidates within a pareto front by crowding distance, less crowded (-> higher value) is preferred """
        for candidate in front:
            candidate.cur_crowding_dist = 0
            candidate.cur_cdh = 0
        for objective in objectives:
            # get indices of a sorted front
            idx_val = sorted([(i, objective.evaluate_candidate(candidate)) for i, candidate in enumerate(front)],
                             key=lambda v: v[1], reverse=objective._maximize)
            sorted_idx, sorted_values = list(zip(*idx_val))
            max_value, min_value = max(sorted_values), min(sorted_values)
            # set all in between, corners and distance between values are preferred
            if max_value > min_value:
                for i in range(1, len(sorted_idx) - 1):
                    front[sorted_idx[i]].cur_cdh += (sorted_values[i - 1] - sorted_values[i + 1]) / (
                                max_value - min_value)
                # set first + last
                front[sorted_idx[0]].cur_cdh = 1
                front[sorted_idx[-1]].cur_cdh = 1
                # add temp var to crowding distance, weighted by objective
                for candidate in front:
                    candidate.cur_crowding_dist += candidate.cur_cdh * objective._weighting
                    candidate.cur_cdh = 0
        front.sort(key=lambda c: c.cur_crowding_dist, reverse=True)

    def order_fronts(self, key: str):
        """ sort all fronts by a metric key """
        for i in range(len(self.fronts)):
            self.fronts[i] = sorted(self.fronts[i], key=lambda c: c.metrics.get(key, 0))

    def filter(self, space: ValueSpace):
        """ remove all candidates that are not in the given ValueSpace """
        allowed = []
        for c in self.candidates:
            if space.is_allowed(c.values):
                allowed.append(c)
        self.candidates = allowed
        self._fronts = []

    def log(self, logger, k=10000):
        """ log all candidates, up to k """
        logger.info('[%s] candidates:' % self.name)
        for i, candidate in enumerate(self.candidates):
            logger.info('  {i:>2} {c}'.format(i=i, c=candidate))
            if i >= k:
                break

    def log_front(self, logger, n=0, k=10000):
        """ log all candidates on the given front number, up to k """
        logger.info('[%s] candidate front [%d]:' % (self.name, n))
        for i, candidate in enumerate(self.fronts[n]):
            logger.info('  {i:>2} {c}'.format(i=i, c=candidate))
            if i >= k:
                break

    def plot(self, key1: str, key2: str, show=True, add_title=False, add_bar=True,
             pareto_lines=1, num_fronts=-1, save_path: str = None):
        """ for (pareto) plots of the population """
        ax = plt.gca()
        max_iter = max([c.iteration for c in self.candidates])
        cmap = plt.get_cmap('plasma')
        for i, front in enumerate(self.fronts):
            pos = []
            if i >= num_fronts > 0:
                break
            for c in front:
                pos.append((c.metrics.get(key1, 0), c.metrics.get(key2, 0)))
                ax.scatter(*pos[-1], c=c.iteration, cmap=cmap, label='__no_legend__', vmin=0, vmax=max_iter, s=16)
            if i < pareto_lines:
                pos = sorted(pos, key=lambda p: p[0])
                ax.plot([p[0] for p in pos], [p[1] for p in pos], '--', c='red', label='__no_legend__')
        if add_title:
            plt.title('Population "%s"' % self.name)
        plt.xlabel(key1)
        plt.ylabel(key2)
        if add_bar and max_iter > 0:
            col_bar = plt.colorbar(ax.get_children()[2], ax=ax)
            col_bar.set_label('iterations')
        if show:
            plt.show()
        if isinstance(save_path, str):
            plt.savefig(save_path)

    @classmethod
    def add_other_pareto_to_plot(cls, population, key1: str, key2: str, show=True, save_path: str = None):
        """  """
        ax = plt.gca()
        pos = []
        for c in population.fronts[0]:
            pos.append((c.metrics.get(key1, 0), c.metrics.get(key2, 0)))
        pos = sorted(pos, key=lambda p: p[0])
        ax.plot([p[0] for p in pos], [p[1] for p in pos], '-', c='green', label='true pareto front')
        if show:
            plt.show()
        if isinstance(save_path, str):
            plt.savefig(save_path)

    @classmethod
    def power_population(cls, populations: list, sum_keys: [str] = None):
        """
        power set across all candidates,
        their values (genes) are put together in order of their respective populations
        """
        pop = Population(name='combined')
        all_combinations = power_list([p.fronts[0] for p in populations])
        for candidates in all_combinations:
            values = []
            keys = dict()
            for c in candidates:
                values.extend(c.values)
                for sk in sum_keys:
                    keys[sk] = keys.get(sk, 0) + c.metrics.get(sk, 0)
            nc = Candidate(values=tuple(values))
            nc.metrics.update(keys)
            pop.append(nc)
        return pop
