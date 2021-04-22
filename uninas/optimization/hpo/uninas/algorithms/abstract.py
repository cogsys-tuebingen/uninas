import pickle
import os
from typing import Union
from uninas.optimization.estimators.abstract import AbstractEstimator
from uninas.optimization.hpo.uninas.candidate import Candidate
from uninas.optimization.hpo.uninas.population import Population
from uninas.optimization.hpo.uninas.values import ValueSpace
from uninas.utils.args import ArgsInterface, Namespace
from uninas.utils.loggers.python import Logger


class AbstractAlgorithm:
    def __init__(self, value_space: ValueSpace, logger: Logger, save_file: Union[str, None]='/tmp/alg.pickle',
                 strategy_name: str = None, constraints=(), estimators_gen=(), estimators_eval=(), objectives=()):
        """
        basic constrained multi-objective optimization algorithm

        :param value_space: space where values can be sampled from
        :param logger: a logger to log to
        :param save_file: to save+load the state of the algorithm, skipped if None
        :param strategy_name: str to specify which weight strategy the architecture weights refer to
        :param constraints: AbstractEstimator evaluated at individual creation, bad individuals are resampled
        :param estimators_gen: AbstractEstimator evaluated at individual creation
        :param estimators_eval: AbstractEstimator evaluated when the population is sorted
        :param objectives: AbstractEstimator evaluated to sort the individuals
        """
        self.value_space = value_space.copy()
        self.logger = logger
        self.__save_file = save_file
        self.strategy_name = strategy_name
        self.constraints = constraints
        self.estimators_gen = list(estimators_gen)
        self.estimators_eval = list(estimators_eval)
        self.objectives = list(objectives)

        # some stats
        self.stats = dict(failed_already_exists=0, failed_value_space=0, failed_constraints=0)

        # to save / restore
        self.population = Population('initial')
        self.candidate_by_values = {}
        self.iteration = 0

    def add_candidate(self, population: Population, candidate: Candidate):
        population.append(candidate)
        self.candidate_by_values[candidate.values] = candidate

    def is_allowed_new_candidate(self, candidate: Candidate) -> bool:
        """ checks whether a candidate has allowed values and is within given constraints """
        if candidate.values in self.candidate_by_values:
            self.stats['failed_already_exists'] += 1
            return False
        if not self.value_space.is_allowed(candidate.values):
            self.stats['failed_value_space'] += 1
            return False
        for constraint in self.constraints:
            constraint.evaluate_candidate(candidate, strategy_name=self.strategy_name)
            if not constraint.is_candidate_allowed(candidate):
                self.stats['failed_constraints'] += 1
                return False
        for estimator in self.estimators_gen:
            estimator.evaluate_candidate(candidate, strategy_name=self.strategy_name)
        return True

    def add_candidate_if_allowed(self, population: Population, candidate: Candidate):
        if self.is_allowed_new_candidate(candidate):
            self.add_candidate(population, candidate)

    def add_random(self, population: Population, num=1, max_iterations=10000):
        """ add random candidates to the pool """
        for i in range(max_iterations):
            candidate = Candidate(values=self.value_space.random_sample(), iteration=self.iteration)
            if self.is_allowed_new_candidate(candidate):
                self.add_candidate(population, candidate)
            if len(population) >= num:
                break

    def _save_file(self, save_file: str = None) -> Union[str, None]:
        return save_file if save_file is not None else self.__save_file

    def save_state(self, save_file: str = None):
        """ save the current search state """
        save_file = self._save_file(save_file)
        if isinstance(save_file, str):
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            if os.path.isfile(save_file):
                os.remove(save_file)
            with open(save_file, 'wb') as file:
                pickle.dump(self._save_dict(), file)
            self.logger.info('Saved %s checkpoint [%s]' % (self.__class__.__name__, save_file))

    def load_state(self, save_file: str = None):
        """ try loading a search state """
        save_file = self._save_file(save_file)
        if isinstance(save_file, str) and os.path.isfile(save_file):
            with open(save_file, 'rb') as file:
                dct = pickle.load(file)
                self._load_dict(dct)
            self.logger.info('Loaded %s checkpoint [%s]' % (self.__class__.__name__, save_file))
        else:
            self.logger.info('Can not load %s checkpoint, file [%s] does not exist' %
                             (self.__class__.__name__, save_file))

    def remove_saved_state(self, save_file: str = None):
        """ remove the saved search state """
        save_file = self._save_file(save_file)
        if isinstance(save_file, str) and os.path.isfile(save_file):
            os.remove(save_file)

    def log_top_k(self, population=None, k=10000, log_all=False, log_all_fronts=False):
        """ log top k candidates in the pool """
        population = population if population is not None else self.population
        if log_all:
            population.log(self.logger, k=k)
        population.log_front(self.logger, n=0, k=k)
        if log_all_fronts:
            for fi in range(1, len(population.fronts)):
                population.log_front(self.logger, n=fi, k=k)

    def get_total_population(self, sort=True, partially=True) -> Population:
        """ a population of all candidates ever evaluated """
        dummy_population = Population(name='result after iteration %s' % self.iteration)
        dummy_population.extend(list(self.candidate_by_values.values()))
        if sort:
            if partially:
                dummy_population.sort_partially_into_fronts(self.objectives, num_dominated=0)
            else:
                dummy_population.sort_into_fronts(self.objectives)
        return dummy_population

    def plot(self, obj1: AbstractEstimator, obj2: AbstractEstimator, pareto_lines=1, show=True, save_path: str = None):
        dummy_population = self.get_total_population(sort=pareto_lines >= 1, partially=pareto_lines <= 1)
        dummy_population.plot(obj1.key, obj2.key, show=show, pareto_lines=pareto_lines, save_path=save_path)

    def search(self, load=True, log_each_iteration=True):
        if load:
            self.load_state()

        self._search(log_each_iteration=log_each_iteration)

        self.logger.info('-'*120)
        self.logger.info('%s search completed!' % self.__class__.__name__)
        self.logger.info('re-sampled candidates before evaluation:')
        self.logger.info('{:>8} candidates were already evaluated'.format(self.stats['failed_already_exists']))
        self.logger.info('{:>8} candidates had invalid values'.format(self.stats['failed_value_space']))
        self.logger.info('{:>8} candidates did not lie within the constraints'.format(self.stats['failed_constraints']))
        self.logger.info('evaluated %d different candidates' % len(self.candidate_by_values))
        self.logger.info('-'*120)
        self.log_top_k(log_all=False, log_all_fronts=False)
        self.save_state()

    def _save_dict(self) -> dict:
        return {
            'population': self.population,
            'candidate_values': self.candidate_by_values,
            'iteration': self.iteration,
            'stats': self.stats,
        }

    def _load_dict(self, dct: dict):
        self.population = dct.get('population')
        self.candidate_by_values = dct.get('candidate_values')
        self.iteration = dct.get('iteration')
        self.stats = dct.get('stats')

    def _search(self, log_each_iteration=True):
        """ search procedure, from generating the first candidate(s) to finding the best ones """
        raise NotImplementedError

    @classmethod
    def save_population(cls, save_file: str, population: Population):
        """ save the population """
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        if os.path.isfile(save_file):
            os.remove(save_file)
        with open(save_file, 'wb') as file:
            pickle.dump(population, file)

    @classmethod
    def load_population(cls, save_file: str) -> Population:
        """ load a population from a save file """
        assert os.path.isfile(save_file)
        with open(save_file, 'rb') as file:
            state = pickle.load(file)
            if isinstance(state, Population):
                return state
            if isinstance(state.get('population'), Population):
                return state.get('population')
            raise ModuleNotFoundError('population not found in %s' % save_file)


class AbstractHPO(ArgsInterface):

    @classmethod
    def run_opt(cls, hparams: Namespace, logger: Logger, checkpoint_dir: str, value_space: ValueSpace,
                constraints: list, objectives: list) -> AbstractAlgorithm:
        """ run the optimization, return all evaluated candidates """
        raise NotImplementedError

    @classmethod
    def is_full_eval(cls) -> bool:
        return False
