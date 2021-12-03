from uninas.tasks.hpo_self import NetHPOTask, SelfHPOUtils
from uninas.methods.strategy_manager import StrategyManager
from uninas.optimization.hpo.uninas.algorithms.randomly import RandomHPO, RandomlyEval
from uninas.optimization.hpo.uninas.population import Population
from uninas.optimization.hpo.uninas.values import DiscreteValues, ValueSpace
from uninas.optimization.estimators.net import NetValueEstimator
from uninas.optimization.task import common_s2_prepare_run
from uninas.utils.args import MetaArgument, Argument
from uninas.utils.loggers.python import log_headline

from uninas.register import Register
from uninas.builder import Builder


@Register.task(search=True)
class DnaHPOTask(NetHPOTask):
    """
    Blockwisely Supervised Neural Architecture Search with Knowledge Distillation
    https://arxiv.org/abs/1911.13053
    https://github.com/changlin31/DNA

    while the original progressively combines the partial architectures and checks constraints,
    we simply check the power set of the architectures, as the costly loss does not have to be evaluated again
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('dna2_max_eval_per_stage', default=5000, type=int, help='max num partial architectures to eval per stage'),
            Argument('dna2_max_eval_final', default=10000, type=int, help='max num full architectures to eval'),
        ]

    @classmethod
    def meta_args_to_add(cls, algorithm=False) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add(algorithm=algorithm)

    def _run(self, save=True):
        common_s2_prepare_run(self.logger, self.trainer, self.s1_path, self.tmp_load_path, self.reset_bn, self.methods)

        eval_per_stage, eval_final = self._parsed_arguments(['dna2_max_eval_per_stage', 'dna2_max_eval_final'], self.args)

        # get the loss estimator
        value_estimator = None
        for o in self.objectives:
            if isinstance(o, NetValueEstimator):
                assert value_estimator is None, "must have exactly one NetValueEstimator for the loss!"
                value_estimator = o
        assert value_estimator is not None, "must have exactly one NetValueEstimator for the loss!"

        # eval each distillation stage/strategy
        sm = StrategyManager()
        strategies = sm.get_strategies()
        populations = []
        for sb in self.get_method().sync_blocks.blocks:
            s = strategies[sb.name]
            checkpoint_dir = self.checkpoint_dir(self.save_dir + '/strategies/%s/' % s.name)
            sm.forward_const(const=0)  # reset arc to only zeros for a fair comparison

            space = ValueSpace(*[DiscreteValues.interval(0, n) for n in s.get_num_choices()])
            space = SelfHPOUtils.mask_architecture_space(self.args, space)

            value_estimator.set_net_kwargs(start_cell=sb.first_teacher_index, end_cell=sb.last_teacher_index)

            algorithm = RandomHPO.run_opt(hparams=self.args, logger=self.logger, checkpoint_dir=checkpoint_dir,
                                          value_space=space, strategy_name=s.name,
                                          constraints=self.constraints, objectives=self.objectives,
                                          num_eval=eval_per_stage)
            populations.append(algorithm.get_total_population(sort=True))

        # merge stages into one
        log_headline(self.logger, 'evaluated each stage, merging results now')
        population = Population.power_population(populations, sum_keys=[value_estimator.key])
        n = len(population)
        population.reduce_to_random_subset(eval_final)
        self.logger.info('Evaluating %d/%d candidates from the combined block search spaces' % (len(population), n))
        population.evaluate(self.constraints)
        population.evaluate(self.objectives)
        population.sort_partially_into_fronts(self.objectives, num_dominated=0)
        population.order_fronts(self.objectives[0].key)
        population.log_front(self.logger)

        # save the final population
        save_file = self.checkpoint_dir(self.save_dir + '/strategies/%s.pickle' % "combined_population")
        RandomlyEval.save_population(save_file, population)

        # save results
        if save:
            checkpoint_dir = self.checkpoint_dir(self.save_dir)
            file_viz = '%sx.pdf' % checkpoint_dir
            candidate_dir = '%s/candidates/' % checkpoint_dir
            population.plot(self.objectives[0].key, self.objectives[1].key, show=False, add_bar=False, save_path=file_viz)
            for candidate in population.fronts[0]:
                self.get_method().get_network().forward_strategy(fixed_arc=candidate.values)
                Builder.save_config(self.get_method().get_network().config(finalize=True),
                                    candidate_dir, 'candidate-%s' % '-'.join([str(g) for g in candidate.values]))

        return None, population
