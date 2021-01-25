"""
NSGA-II (non-dominant sorting genetic algorithm 2)
https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf

Principles:
    1) elitism, the best individuals get a chance to reproduce
    2) crowding distance, preserving diversity in the population
    3) emphasizing non-dominated solutions

Process:
    1) generate candidates in the gene space until the population is full, checking 2)
    2) evaluate generated candidates
        2.1) check if they are within the gene space and given constraints
        2.2) evaluate optional additional metrics (estimators_gen)
    3) evaluate generated candidates, now that the population is fixed for this step/epoch/iteration
        3.1) evaluate optional additional metrics (estimators_eval)
        3.2) evaluate objectives
    4) sort population
        4.1) build pareto fronts according to objectives
    5) select best individuals for the next generation
        5.1) pick all fronts that fully fit
        5.2) for the partially fitting front, first sort by crowding distance, then take the least crowded candidates
    6) stop now if we've exceeded the search budget (number of epochs/iterations)
    7) expand population
        7.1) crossover
        7.2) mutation
    8) back to 2)

future ideas
- use NSGA2 to search hyperparams for training
    - queue jobs in slurmer, e.g. via class SlurmerQueue(AbstractEstimator)
    - fetch results via SlurmerRead(AbstractEstimator), need to be able to wait until slurm jobs finished
    - ContinuousValues(AbstractValues), avoid sampling new hyperparam too close to the current one
"""

import random
from collections.abc import Iterable
from uninas.optimization.hpo.uninas.algorithms.abstract import AbstractAlgorithm, AbstractHPO
from uninas.optimization.hpo.uninas.values import AbstractValues, ValueSpace
from uninas.optimization.hpo.uninas.candidate import Candidate
from uninas.optimization.hpo.uninas.crossover import Crossover, SinglePointCrossover, MixedCrossover
from uninas.optimization.hpo.uninas.population import Population
from uninas.utils.args import Argument, Namespace
from uninas.utils.loggers.python import log_headline, Logger
from uninas.register import Register


class NSGA2(AbstractAlgorithm):
    def __init__(self, value_space: [AbstractValues], crossover: Crossover,
                 logger: Logger, save_file='/tmp/nsga2.pickle', strategy_name: str = None,
                 max_iterations=10, population_size=50, population_core_size=10,
                 num_tourney_participants=5, mutation_probability=1,
                 constraints=(), estimators_gen=(), estimators_eval=(), objectives=()):
        """
        NSGA2 genetic optimization

        :param value_space: space where genes can be sampled from
        :param logger: a logger to log to
        :param crossover: class to handle crossover between two individuals
        :param save_file: to save+load the state of the algorithm
        :param strategy_name: str to specify which weight strategy the architecture weights refer to
        :param max_iterations:
        :param population_size: maximum size that the population will have after expanding
        :param population_core_size: minimum size that the population has after selection
        :param num_tourney_participants:
        :param mutation_probability: probability for each gene to be randomized, after crossover
        :param constraints: AbstractEstimator evaluated at individual creation, bad individuals are resampled
        :param estimators_gen: AbstractEstimator evaluated at individual creation
        :param estimators_eval: AbstractEstimator evaluated when the population is sorted
        :param objectives: AbstractEstimator evaluated to sort the individuals
        """
        super().__init__(value_space=value_space, logger=logger, save_file=save_file, strategy_name=strategy_name,
                         constraints=constraints, estimators_gen=estimators_gen,
                         estimators_eval=estimators_eval, objectives=objectives)
        self.crossover = crossover
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.population_core_size = population_core_size
        self.num_tourney_participants = num_tourney_participants
        self.mutation_probability = mutation_probability

    def add_children(self, population: Population, num=1, max_iterations=10000):
        """ add children of the pool to the pool """
        children, children_genes = [], {}
        for candidate in self._mutated_children_generator(population, max_iterations=max_iterations):
            if candidate.values not in children_genes and self.is_allowed_new_candidate(candidate):
                children.append(candidate)
                children_genes[candidate.values] = True
            if len(children) >= num:
                break
        for c in children:
            self.add_candidate(population, c)
        if len(children) < num:
            self.logger.warning('Added fewer children than requested (%d/%d), stopped after %d attempts.' %
                                (len(children), num, max_iterations))

    def _mutated_children_generator(self, population: Population, max_iterations=10000) -> Iterable:
        """ generate and mutate crossover children (candidates) of the population """
        for gene in self._gene_generator(population, max_iterations):
            yield Candidate(values=tuple(self._mutate(gene)), iteration=self.iteration)

    def _gene_generator(self, population: Population, max_iterations=10000) -> Iterable:
        """ generate crossover genes (lists) of the population """
        for i in range(max_iterations):
            p0 = population.random_tourney_sample(num_participants=self.num_tourney_participants)
            p1 = population.random_tourney_sample(num_participants=self.num_tourney_participants)
            for r in self.crossover.yield_genes(p0, p1):
                yield r

    def _mutate(self, gene: list) -> list:
        for i, g in enumerate(self.value_space):
            if random.random() < self.mutation_probability:
                gene[i] = g.sample(prev=gene[i])
        return gene

    def _search(self, log_each_iteration=True):
        """ search procedure, from generating the first candidate(s) to finding the best ones """
        if self.iteration == 0:
            # fill the population with random candidates, evaluate and sort them
            self.add_random(self.population, num=self.population_size)
            self.population.evaluate(self.estimators_eval + self.objectives, strategy_name=self.strategy_name)
            self.population.sort_into_fronts(self.objectives)
            self.iteration += 1
            self.save_state()

        while self.iteration < self.max_iterations:
            if log_each_iteration:
                self.log_top_k(log_all=False, log_all_fronts=False)

            # create the next population from the currently best candidates
            next_population = Population('iteration %d' % self.iteration)
            for i, front in enumerate(self.population.fronts):
                Population.sort_within_front(front, self.objectives)
                rem = self.population_core_size - (len(next_population) + len(front))
                if rem >= 0:
                    next_population.extend(front)
                    continue
                elif len(front)+rem > 0:
                    next_population.extend(front[:-abs(rem)])
                break
            # extend the population
            self.add_children(next_population, self.population_size - len(next_population))
            # evaluate and sort the population
            self.population = next_population
            self.population.evaluate(self.estimators_eval + self.objectives, strategy_name=self.strategy_name)
            self.population.sort_into_fronts(self.objectives)
            self.population.order_fronts(self.objectives[0].key)
            self.iteration += 1
            self.save_state()


@Register.hpo_self_algorithm()
class NSGA2HPO(AbstractHPO):
    """
    NSGA-II (non-dominant sorting genetic algorithm 2)
    https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf

    Principles:
        1) elitism, the best individuals get a chance to reproduce
        2) crowding distance, preserving diversity in the population
        3) emphasizing non-dominated solutions
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('iterations', default=5, type=int, help='maximum number of iterations/epochs'),
            Argument('pop_size', default=5, type=int, help='maximum population size after crossover'),
            Argument('pop_core', default=2, type=int, help='minimum population size after selection'),
            Argument('num_tourney', default=2, type=int, help='num participants in tourney selection'),
            Argument('prob_mutations', default=0.1, type=float, help='probability for mutations'),
            Argument('crossover', default='single', type=str, help='crossover type', choices=['single', 'mixed']),
        ]

    @classmethod
    def run_opt(cls, hparams: Namespace, logger: Logger, checkpoint_dir: str, value_space: ValueSpace,
                constraints: list, objectives: list) -> AbstractAlgorithm:
        """ run the optimization, return all evaluated candidates """
        crossover = {
            'single': SinglePointCrossover,
            'mixed': MixedCrossover,
        }[cls._parsed_argument('crossover', hparams)](value_space)

        nsga2 = NSGA2(
            value_space=value_space,
            crossover=crossover,
            logger=logger,
            save_file='%s/%s.pickle' % (checkpoint_dir, NSGA2.__name__),
            max_iterations=cls._parsed_argument('iterations', hparams),
            population_size=cls._parsed_argument('pop_size', hparams),
            population_core_size=cls._parsed_argument('pop_core', hparams),
            num_tourney_participants=cls._parsed_argument('num_tourney', hparams),
            mutation_probability=cls._parsed_argument('prob_mutations', hparams),
            constraints=constraints,
            objectives=objectives)

        # load, search, return
        log_headline(logger, 'Starting %s' % cls.__name__)
        nsga2.search(load=True, log_each_iteration=True)
        return nsga2
