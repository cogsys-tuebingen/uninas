import os
import shutil
from collections import defaultdict
from copy import deepcopy
from uninas.tasks.abstract import AbstractTask, AbstractNetTask
from uninas.optimization.common.task import common_s2_net_args_to_add, common_s2_extend_args, common_s2_prepare_run
from uninas.optimization.hpo_self.algorithms.abstract import AbstractHPO
from uninas.optimization.hpo_self.algorithms.randomly import RandomlyEval
from uninas.optimization.hpo_self.values import DiscreteValues, ValueSpace, SpecificValueSpace
from uninas.utils.paths import replace_standard_paths
from uninas.utils.misc import split
from uninas.utils.args import MetaArgument, Argument, Namespace, find_in_args
from uninas.utils.loggers.python import log_headline, Logger
from uninas.benchmarks.mini import MiniNASBenchApi
from uninas.register import Register
from uninas.builder import Builder


class SelfHPOUtils:
    """
    shared parts
    """

    @staticmethod
    def prepare(cls: AbstractTask.__class__, logger: Logger, estimator_kwargs: dict, args: Namespace, index=None)\
            -> (AbstractHPO, [], []):
        """
        :param cls:
        :param logger:
        :param estimator_kwargs:
        :param args: global namespace
        :param index: index of the task
        :return: hpo class, constraints, objectives
        """

        # hp optimizer
        try:
            hpo = cls._parsed_meta_argument('cls_hpo_self_algorithm', args, index=index)
            assert issubclass(hpo, AbstractHPO), 'Method must have class methods to optimize the arc'
        except:
            hpo = None

        # estimators
        log_headline(logger, 'adding network estimators')
        constraints, objectives = [], []
        for i, e in enumerate(cls._parsed_meta_arguments('cls_hpo_estimators', args, index=index)):
            estimator = e(args=args, index=i, **estimator_kwargs)
            if estimator.is_constraint():
                constraints.append(estimator)
            if estimator.is_objective():
                objectives.append(estimator)
            logger.info(estimator.str())
        return hpo, constraints, objectives

    @classmethod
    def meta_args_to_add(cls, estimators=True, algorithm=True) -> [MetaArgument]:
        meta = []
        if estimators:
            meta.append(MetaArgument('cls_hpo_estimators', Register.hpo_estimators, help_name='estimator', allow_duplicates=True, allowed_num=(1, -1)))
        if algorithm:
            meta.append(MetaArgument('cls_hpo_self_algorithm', Register.hpo_self_algorithms, help_name='hyper-parameter optimizer', allowed_num=1))
        return meta

    @staticmethod
    def mask_architecture_space(args: Namespace, space: ValueSpace) -> ValueSpace:
        _, mask = find_in_args(args, ".mask_indices")
        for i in split(mask, int):
            space.remove_value(i)
        return space


@Register.task(search=True)
class MiniBenchHPOTask(AbstractTask):
    """
    A hyper-parameter optimization task without networks/methods, purely on a given mini-bench
    """

    def __init__(self, args: Namespace, wildcards: dict):
        AbstractTask.__init__(self, args, wildcards)
        self.mini_bench = MiniNASBenchApi.load(self._parsed_argument('mini_bench_path', args))
        self.mini_bench_dataset = self._parsed_argument('mini_bench_dataset', args)
        self.plot_true_pareto = self._parsed_argument('plot_true_pareto', args)

        estimator_kwargs = dict(mini_api=self.mini_bench, mini_api_set=self.mini_bench_dataset)
        self.hpo, self.constraints, self.objectives = SelfHPOUtils.prepare(self, self.logger, estimator_kwargs, args)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return AbstractTask.args_to_add(index) + [
            Argument('mini_bench_path', default='{path_data}/mini.pt', type=str, help='', is_path=True),
            Argument('mini_bench_dataset', default='cifar10', type=str, help=''),
            Argument('plot_true_pareto', default='False', type=str, help='add the true pareto front', is_bool=True),
            Argument('mask_indices', default="", type=str, help='[int] mask specific primitives from being used'),
        ]

    @classmethod
    def meta_args_to_add(cls, **_) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + SelfHPOUtils.meta_args_to_add(estimators=True, algorithm=True)

    def _run(self):
        file_viz = '%s/%s.pdf' % (self.checkpoint_dir(self.save_dir), self.hpo.__name__)
        space = SelfHPOUtils.mask_architecture_space(self.args, self.mini_bench.get_space())
        algorithm = self.hpo.run_opt(hparams=self.args, logger=self.logger,
                                     checkpoint_dir=self.checkpoint_dir(self.save_dir),
                                     value_space=space,
                                     constraints=self.constraints, objectives=self.objectives)
        population = algorithm.get_total_population(sort=True)
        population.plot(self.objectives[0].key, self.objectives[1].key, show=False, save_path=file_viz, num_fronts=-1)

        if self.plot_true_pareto and not self.hpo.is_full_eval():
            log_headline(self.logger, 'Starting a full evaluation to get the true pareto front')
            full = RandomlyEval(
                value_space=space,
                logger=self.logger,
                save_file='%s/%s.pickle' % (self.checkpoint_dir(self.save_dir), RandomlyEval.__name__),
                constraints=self.constraints,
                objectives=self.objectives,
                num_eval=-1)
            full.search(load=True)
            population.add_other_pareto_to_plot(full.population, self.objectives[0].key, self.objectives[1].key,
                                                show=False, save_path=file_viz)
        return algorithm, population


@Register.task(search=True)
class NetHPOTask(AbstractNetTask):
    """
    An s2 task (trying to figure out the optimal network architecture of a trained s1 network)
    the chosen method contains the exact optimization approach
    """

    def __init__(self, args: Namespace, wildcards: dict):
        AbstractNetTask.__init__(self, args, wildcards)

        # args
        self.reset_bn = self._parsed_argument('reset_bn', args)
        self.s1_path = replace_standard_paths(self._parsed_argument('s1_path', args))

        # files
        self.tmp_load_path = '%s/checkpoint.tmp.pt' % self.save_dir
        os.makedirs(os.path.dirname(self.tmp_load_path), exist_ok=True)
        shutil.copyfile('%s/data.meta.pt' % self.s1_path, '%s/data.meta.pt' % self.save_dir)

        # one method, one trainer... could be executed in parallel in future?
        log_headline(self.logger, 'adding Method, Trainer, ...')
        self.add_method()
        self.add_trainer(method=self.methods[0], save_dir=self.save_dir, num_devices=-1)
        self.log_methods_and_trainer()
        self.methods[0].get_network().set_forward_strategy(False)

        # algorithms
        estimator_kwargs = dict(trainer=self.trainer[0], load_path=self.tmp_load_path)
        self.hpo, self.constraints, self.objectives = SelfHPOUtils.prepare(self, self.logger, estimator_kwargs, args)

        # arc space
        space = ValueSpace(*[DiscreteValues.interval(0, n) for n in self.methods[0].strategy.ordered_num_choices(unique=True)])
        self._architecture_space = SelfHPOUtils.mask_architecture_space(self.args, space)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + common_s2_net_args_to_add()

    @classmethod
    def meta_args_to_add(cls, estimators=True, algorithm=True) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + SelfHPOUtils.meta_args_to_add(estimators=estimators, algorithm=algorithm)

    @classmethod
    def extend_args(cls, args_list: [str]):
        """
        allow modifying the arguments list before other classes' arguments are dynamically added
        this should be used sparsely, as it is hard to keep track of
        """
        common_s2_extend_args(cls, args_list)

    def _run(self, save=True):
        common_s2_prepare_run(self.logger, self.trainer, self.s1_path, self.tmp_load_path, self.reset_bn, self.methods)
        checkpoint_dir = self.checkpoint_dir(self.save_dir)
        file_candidate = '%s/candidates/%s.network_config' % (checkpoint_dir, 'candidate-%s')
        file_viz = '%sx.pdf' % checkpoint_dir
        self.methods[0].eval()

        # run
        algorithm = self.hpo.run_opt(hparams=self.args, logger=self.logger, checkpoint_dir=checkpoint_dir,
                                     value_space=self._architecture_space,
                                     constraints=self.constraints, objectives=self.objectives)
        population = algorithm.get_total_population(sort=True)

        # save results
        if save:
            population.plot(self.objectives[0].key, self.objectives[1].key, show=False, save_path=file_viz)
            for candidate in population.fronts[0]:
                self.methods[0].get_network().forward_strategy(fixed_arc=candidate.values)
                Builder.save_config(self.methods[0].get_network().config(finalize=True),
                                    file_candidate % '-'.join([str(g) for g in candidate.values]))
        return algorithm, population


@Register.task(search=True)
class EvalNetBenchTask(NetHPOTask):
    """
    Evaluate a s1 network and immediately compare results with a bench
    """

    def __init__(self, args: Namespace, wildcards: dict):
        super().__init__(args, wildcards)

        # restrictions
        assert len(self.objectives) == 1

        # bench part
        self.mini_bench = MiniNASBenchApi.load(self._parsed_argument('mini_bench_path', args))
        self.mini_bench_dataset = self._parsed_argument('mini_bench_dataset', args)
        self.measure_top = split(self._parsed_argument('measure_top', self.args), int)
        # check if the cell architecture was shared during training
        self.num_normal = 1
        _, arc_shared = find_in_args(self.args, '.arc_shared')
        if not arc_shared:
            _, cell_order = find_in_args(self.args, '.cell_order')
            self.num_normal = cell_order.count('n')

        # correlations
        self.correlation_cls = []
        for name in split(self._parsed_argument('measure_correlations', self.args)):
            self.correlation_cls.append(Register.get(name))

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('mini_bench_path', default='{path_data}/mini.pt', type=str, help='', is_path=True),
            Argument('mini_bench_dataset', default='cifar10', type=str, help='data set to rank topN networks on'),
            Argument('measure_top', default='10, 50', type=str, help='measure topN bench architectures'),
            Argument('measure_correlations', default='KendallTauCorrelation', type=str, help='correlations to measure'),
        ]

    def _run(self, save=False):
        checkpoint_dir = self.checkpoint_dir(self.save_dir)
        file_plot = '%s/plots/%s/%s/%s.pdf' % (checkpoint_dir, '%s', '%s', '%s')

        # specific sorted value space
        svs = [v.arch_tuple for v in self.mini_bench.get_all_sorted('acc1', self.mini_bench_dataset, reverse=True)]
        arc_len = len(svs[0])
        if self.num_normal > 1:
            # compensate now for late architecture sharing by duplicating the indices
            svs = [tuple(list(v)*self.num_normal) for v in svs]
        self._svs = SelfHPOUtils.mask_architecture_space(self.args, SpecificValueSpace(svs))

        algorithm, name_num_rem = None, [(str(v), v, False) for v in self.measure_top] + [('all', 9999999999, True)]
        for name, num, rem in name_num_rem:
            if algorithm is not None and rem:
                algorithm.remove_saved_state()

            self._architecture_space = deepcopy(self._svs)
            self._architecture_space.specific_values = self._architecture_space.specific_values[:num]
            algorithm, population = super()._run(save=save)

            # compare
            key = self.objectives[0].key
            net_values = []
            bench_values = defaultdict(list)
            for candidate in population.candidates:
                net_values.append(candidate.metrics.get(key))
                r = self.mini_bench.get_by_arch_tuple(candidate.values[:arc_len])
                for k, v in r.named_dicts().get(key).items():
                    bench_values[k].append(v)

            # plots, tb
            self.get_first_method().log_metrics({
                'net_bench/%s/num' % name: population.size
            })
            for k, bv in bench_values.items():
                # generate plot
                m = self.correlation_cls[0](column_names=("network", "bench %s" % self.mini_bench.bench_type),
                                            add_lines=False, can_show=False)
                m.add_data(net_values, bv, "%s, %s" % (k, key), other_metrics=self.correlation_cls, s=8)
                m.plot(legend=True, show=False, save_path=file_plot % ('metrics', name, k))

                # log general things
                self.get_first_method().log_metrics({
                    'net_bench/%s/%s/%s/avg/bench' % (name, key, k): sum(bv) / len(bv),
                    'net_bench/%s/%s/%s/avg/net' % (name, key, k): sum(net_values) / len(net_values),
                })

                # log metrics
                for m in self.correlation_cls:
                    self.get_first_method().log_metrics({
                        'net_bench/%s/%s/%s/%s' % (name, key, k, m.short_name()): m.calculate(net_values, bv),
                    })
