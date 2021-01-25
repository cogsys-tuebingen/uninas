"""
pure random, without successive halving etc.
"""

import numpy as np
from uninas.optimization.hpo.uninas.algorithms.abstract import AbstractAlgorithm
from uninas.optimization.hpo.uninas.candidate import Candidate
from uninas.optimization.hpo.uninas.algorithms.abstract import AbstractHPO
from uninas.optimization.hpo.uninas.values import ValueSpace, AbstractValues
from uninas.utils.args import Argument, Namespace
from uninas.utils.loggers.python import log_headline, Logger
from uninas.register import Register


class RandomlyEval(AbstractAlgorithm):

    def __init__(self, value_space: [AbstractValues],
                 logger: Logger, save_file='/tmp/random.pickle', strategy_name: str = None,
                 constraints=(), estimators_gen=(), estimators_eval=(), objectives=(),
                 num_eval=100):
        """
        Evaluate the entire search space totally randomly

        :param value_space: space where genes can be sampled from
        :param logger: a logger to log to
        :param save_file: to save+load the state of the algorithm
        :param strategy_name: str to specify which weight strategy the architecture weights refer to
        :param constraints: AbstractEstimator evaluated at individual creation, bad individuals are resampled
        :param estimators_gen: AbstractEstimator evaluated at individual creation
        :param estimators_eval: AbstractEstimator evaluated when the population is sorted
        :param objectives: AbstractEstimator evaluated to sort the individuals
        :param num_eval: number of sampled models, all if < 0
        """
        super().__init__(value_space=value_space, logger=logger, save_file=save_file, strategy_name=strategy_name,
                         constraints=constraints, estimators_gen=estimators_gen,
                         estimators_eval=estimators_eval, objectives=objectives)
        self.num_eval = num_eval

    def _search(self, log_each_iteration=True):
        """ search procedure, from generating the first candidate(s) to finding the best ones """
        if self.value_space.is_discrete():
            all_values = [v for v in self.value_space.iterate()]
            np.random.shuffle(all_values)
            for values in all_values:
                if self.population.size >= self.num_eval > 0:
                    break
                self.add_candidate_if_allowed(self.population, Candidate(values=values))
            self.logger.info('added %d/%d candidates, evaluating' % (self.population.size, len(all_values)))
        else:
            self.add_random(self.population, num=max(self.num_eval, 1))
            self.logger.info('added %d candidates, evaluating' % self.population.size)

        self.population.evaluate(self.estimators_eval + self.objectives, strategy_name=self.strategy_name)
        self.population.sort_partially_into_fronts(self.objectives, num_dominated=0)
        self.population.order_fronts(self.objectives[0].key)


@Register.hpo_self_algorithm()
class RandomHPO(AbstractHPO):
    """
    Pure random, without successive halving etc.
    If a candidate is not within the constraints, evaluate another instead
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('num_eval', default=100, type=int, help='number of candidates to eval'),
        ]

    @classmethod
    def run_opt(cls, hparams: Namespace, logger: Logger, checkpoint_dir: str, value_space: ValueSpace,
                constraints: list, objectives: list, num_eval=None, strategy_name=None) -> AbstractAlgorithm:
        """ run the optimization, return all evaluated candidates """

        randomly = RandomlyEval(
            value_space=value_space,
            logger=logger,
            save_file='%s/%s.pickle' % (checkpoint_dir, RandomlyEval.__name__),
            constraints=constraints,
            objectives=objectives,
            num_eval=num_eval if num_eval is not None else cls._parsed_argument('num_eval', hparams),
            strategy_name=strategy_name)

        # load, search, return
        log_headline(logger, 'Starting %s' % cls.__name__)
        randomly.search(load=True, log_each_iteration=True)
        return randomly
