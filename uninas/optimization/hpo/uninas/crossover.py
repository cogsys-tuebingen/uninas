import random
from collections.abc import Iterable
from uninas.optimization.hpo.uninas.candidate import Candidate
from uninas.optimization.hpo.uninas.values import ValueSpace


class Crossover:
    def __init__(self, value_space: ValueSpace, fixed_num_crossover: int = None):
        self.value_space_size = value_space.num_choices()
        self.fixed_num_crossover = fixed_num_crossover

    def _num_crossover(self) -> int:
        if self.fixed_num_crossover is not None:
            return self.fixed_num_crossover
        return random.randint(1, self.value_space_size - 1)

    def yield_genes(self, c0: Candidate, c1: Candidate) -> Iterable:
        """ yield lists (genes) """
        raise NotImplementedError


class MixedCrossover(Crossover):
    """ take genes randomly from either candidate """

    def yield_genes(self, c0: Candidate, c1: Candidate) -> Iterable:
        mask = random.sample(range(self.value_space_size), k=self._num_crossover())
        new_gene0, new_gene1 = [], []
        for j, (gene0, gene1) in enumerate(zip(c0.values, c1.values)):
            g0_, g1_ = (gene0, gene1) if j in mask else (gene1, gene0)
            new_gene0.append(g0_)
            new_gene1.append(g1_)
        yield new_gene0
        yield new_gene1


class SinglePointCrossover(Crossover):
    """ take the first n genes from the first candidate, the rest from the second """

    def yield_genes(self, c0: Candidate, c1: Candidate) -> Iterable:
        n = self._num_crossover()
        yield list(c0.values[:n] + c1.values[n:])
        yield list(c1.values[:n] + c0.values[n:])
