"""
evaluate only the specified topologies
"""

from uninas.optimization.hpo.uninas.algorithms.abstract import AbstractAlgorithm
from uninas.optimization.hpo.uninas.candidate import Candidate
from uninas.optimization.hpo.uninas.algorithms.abstract import AbstractHPO
from uninas.optimization.hpo.uninas.values import ValueSpace, SpecificValueSpace
from uninas.utils.args import Argument, Namespace
from uninas.utils.misc import split
from uninas.utils.loggers.python import log_headline, Logger
from uninas.register import Register


class SpecificEval(AbstractAlgorithm):

    def _search(self, log_each_iteration=True):
        """ search procedure, from generating the first candidate(s) to finding the best ones """
        for v in self.value_space.iterate():
            self.add_candidate_if_allowed(self.population, Candidate(values=v))

        self.population.evaluate(self.estimators_eval + self.objectives, strategy_name=self.strategy_name)
        self.population.sort_partially_into_fronts(self.objectives, num_dominated=0)
        self.population.order_fronts(self.objectives[0].key)


@Register.hpo_self_algorithm()
class SpecificHPO(AbstractHPO):
    """
    Pure random, without successive halving etc.
    If a candidate is not within the constraints, evaluate another instead
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('values', default="", type=str,
                     help='specific values to evaluate:'
                          'use commas to separate those within a group and semicolons to separate groups.'
                          'use -1 as a wildcard to specify all candidates that only differ in these values.'),
        ]

    @classmethod
    def run_opt(cls, hparams: Namespace, logger: Logger, checkpoint_dir: str, value_space: ValueSpace,
                constraints: list, objectives: list, num_eval=None, strategy_name=None) -> AbstractAlgorithm:
        """ run the optimization, return all evaluated candidates """

        # make sure all specified values are in the space, filter the value space to only contain those then
        candidates = []
        for value_group in cls._parsed_argument('values', hparams).replace(' ', '').replace('x', '-1').split(';'):
            value_group = tuple(split(value_group, int))
            if len(value_group) > 0:
                assert value_space.is_allowed(value_group), "Candidate %s is not in the search space" % str(value_group)
                for v in value_space.iterate(value_group):
                    candidates.append(v)
        if isinstance(num_eval, int):
            candidates = candidates[:num_eval]

        specific = SpecificEval(
            value_space=SpecificValueSpace(candidates),
            logger=logger,
            save_file='%s/%s.pickle' % (checkpoint_dir, SpecificEval.__name__),
            constraints=constraints,
            objectives=objectives,
            strategy_name=strategy_name)

        # load, search, return
        log_headline(logger, 'Starting %s' % cls.__name__)
        specific.search(load=True, log_each_iteration=True)

        logger.info("All evaluated candidates")
        for candidate in specific.get_total_population(sort=True):
            logger.info(" > %s" % str(candidate))

        return specific

