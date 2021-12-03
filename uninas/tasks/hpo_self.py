import os
import shutil
from collections import defaultdict
from copy import deepcopy
import numpy as np
from uninas.tasks.abstract import AbstractTask, AbstractNetTask
from uninas.models.networks.uninas.search import SearchUninasNetwork
from uninas.methods.strategy_manager import StrategyManager
from uninas.optimization.task import common_s2_net_args_to_add, common_s2_extend_args, common_s2_prepare_run
from uninas.optimization.benchmarks.mini.benchmark import MiniNASBenchmark, MiniResult
from uninas.optimization.benchmarks.mini.tabular import MiniNASTabularBenchmark, explore
from uninas.optimization.benchmarks.mini.tabular_search import MiniNASSearchTabularBenchmark
from uninas.optimization.hpo.uninas.algorithms.abstract import AbstractHPO
from uninas.optimization.hpo.uninas.algorithms.randomly import RandomlyEval
from uninas.optimization.hpo.uninas.values import DiscreteValues, ValueSpace, SpecificValueSpace
from uninas.utils.paths import replace_standard_paths
from uninas.utils.misc import split
from uninas.utils.args import MetaArgument, Argument, Namespace, find_in_args
from uninas.utils.loggers.python import log_headline, Logger
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
            hpo = cls._parsed_meta_argument(Register.hpo_self_algorithms, 'cls_hpo_self_algorithm', args, index=index)
            assert issubclass(hpo, AbstractHPO), 'Method must have class methods to optimize the arc'
        except:
            hpo = None

        # estimators
        log_headline(logger, 'adding network estimators')
        constraints, objectives = [], []
        for i, e in enumerate(cls._parsed_meta_arguments(Register.hpo_estimators, 'cls_hpo_estimators', args, index=index)):
            estimator = e(args=args, index=i, **estimator_kwargs)
            if estimator.is_constraint():
                constraints.append(estimator)
            if estimator.is_objective():
                objectives.append(estimator)
            logger.info(estimator.str())
        return hpo, constraints, objectives

    @classmethod
    def meta_args_to_add(cls, estimator_filter: dict = None, algorithm=True) -> [MetaArgument]:
        estimators = Register.hpo_estimators
        if isinstance(estimator_filter, dict):
            estimators = estimators.filter_match_all(**estimator_filter)
        meta = [MetaArgument('cls_hpo_estimators', estimators, help_name='estimator', allow_duplicates=True, allowed_num=(1, -1))]
        if algorithm:
            meta.append(MetaArgument('cls_hpo_self_algorithm', Register.hpo_self_algorithms, help_name='hyper-parameter optimizer', allowed_num=1))
        return meta

    @staticmethod
    def mask_architecture_space(args: Namespace, space: ValueSpace) -> ValueSpace:
        _, mask = find_in_args(args, ".mask_indices")
        for i in split(mask, int):
            space.remove_value(i)
        return space

    @staticmethod
    def bench_subspace(args: Namespace, bench: MiniNASTabularBenchmark) -> MiniNASTabularBenchmark:
        _, mask = find_in_args(args, ".mask_indices")
        masked = [i for i in split(mask, int)]
        return bench.subset(blacklist=masked)


@Register.task(search=True)
class MiniBenchHPOTask(AbstractTask):
    """
    A hyper-parameter optimization task without networks/methods, purely on a given mini-bench
    """

    def __init__(self, args: Namespace, *args_, **kwargs):
        super().__init__(args, *args_, **kwargs)
        benchmark_set = self._parsed_meta_argument(Register.benchmark_sets, 'cls_benchmark', args, index=None)
        self.benchmark_set = benchmark_set.from_args(args, index=None)
        self.plot_true_pareto = self._parsed_argument('plot_true_pareto', args)

        estimator_kwargs = dict(mini_api=self.benchmark_set)
        self.hpo, self.constraints, self.objectives = SelfHPOUtils.prepare(self, self.logger, estimator_kwargs, args)

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        benchmark_sets = Register.benchmark_sets.filter_match_all(mini=True)
        return super().meta_args_to_add() + [
            MetaArgument('cls_benchmark', benchmark_sets, allowed_num=1, help_name='mini benchmark set to optimize on'),
        ] + SelfHPOUtils.meta_args_to_add(algorithm=True, estimator_filter=dict(requires_bench=True))

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('plot_true_pareto', default='False', type=str, help='add the true pareto front', is_bool=True),
            Argument('mask_indices', default="", type=str, help='[int] mask specific primitives from being used'),
        ]

    def _run(self):
        file_viz = '%s/%s.pdf' % (self.checkpoint_dir(self.save_dir), self.hpo.__name__)
        space = SelfHPOUtils.bench_subspace(self.args, self.benchmark_set).get_value_space()
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

    def __init__(self, args: Namespace, *args_, **kwargs):
        AbstractNetTask.__init__(self, args, *args_, **kwargs)

        # args
        self.reset_bn = self._parsed_argument('reset_bn', args)
        self.s1_path = replace_standard_paths(self._parsed_argument('s1_path', args))

        # files
        self.tmp_load_path = '%s/checkpoint.tmp.pt' % self.save_dir
        os.makedirs(os.path.dirname(self.tmp_load_path), exist_ok=True)
        shutil.copyfile('%s/data.meta.pt' % self.s1_path, '%s/data.meta.pt' % self.save_dir)

        # one method, one trainer... could be executed in parallel in future?
        log_headline(self.logger, 'setting up...')
        self.add_method()
        self.add_trainer(method=self.get_method(), save_dir=self.save_dir, num_devices=-1)
        self.log_detailed()
        self.get_method().get_network().set_forward_strategy(False)

        # algorithms
        estimator_kwargs = dict(trainer=self.trainer[0], load_path=self.tmp_load_path)
        self.hpo, self.constraints, self.objectives = SelfHPOUtils.prepare(self, self.logger, estimator_kwargs, args)

        # arc space
        space = ValueSpace(*[DiscreteValues.interval(0, n) for n in self.get_method().strategy_manager.get_num_choices(unique=True)])
        self._architecture_space = SelfHPOUtils.mask_architecture_space(self.args, space)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + common_s2_net_args_to_add()

    @classmethod
    def meta_args_to_add(cls, algorithm=True) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + SelfHPOUtils.meta_args_to_add(algorithm=algorithm, estimator_filter=dict(requires_bench=False))

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
        candidate_dir = '%s/candidates/' % checkpoint_dir
        file_viz = '%sx.pdf' % checkpoint_dir
        self.get_method().eval()

        # run
        algorithm = self.hpo.run_opt(hparams=self.args, logger=self.logger, checkpoint_dir=checkpoint_dir,
                                     value_space=self._architecture_space,
                                     constraints=self.constraints, objectives=self.objectives)
        population = algorithm.get_total_population(sort=True)

        # save results
        if save:
            population.plot(self.objectives[0].key, self.objectives[1].key, show=False, save_path=file_viz)
            for candidate in population.fronts[0]:
                self.get_method().get_network().forward_strategy(fixed_arc=candidate.values)
                Builder.save_config(self.get_method().get_network().config(finalize=True),
                                    candidate_dir, 'candidate-%s' % '-'.join([str(g) for g in candidate.values]))
        return algorithm, population


@Register.task(search=True)
class EvalBenchTask(AbstractTask):
    """
    Correlating two or more benches with each other
    """

    def __init__(self, args: Namespace, *args_, **kwargs):
        AbstractTask.__init__(self, args, *args_, **kwargs)

        # benchmarks
        self.same_dataset = self._parsed_argument("same_dataset", args, index=None)
        benchmark_sets = self._parsed_meta_arguments(Register.benchmark_sets, 'cls_benchmarks', args, index=None)
        self.benchmark_sets = [bs.from_args(args, index=i) for i, bs in enumerate(benchmark_sets)]

        # correlations
        self.correlation_cls = []
        for name in self._parsed_argument('measure_correlations', self.args, split_=True):
            self.correlation_cls.append(Register.nas_metrics.get(name))

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        benchmark_sets = Register.benchmark_sets.filter_match_all(mini=True, tabular=True)
        return super().meta_args_to_add() + [
            MetaArgument('cls_benchmarks', benchmark_sets, allowed_num=(2, -1), allow_duplicates=True,
                         help_name='mini benchmark sets to correlate with each other'),
        ] + SelfHPOUtils.meta_args_to_add(algorithm=True, estimator_filter=dict(requires_bench=True))

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('same_dataset', default='False', type=str, is_bool=True,
                     help="correlate only if the bench results are on the same dataset"),
            Argument('measure_correlations', default='KendallTauNasMetric', type=str, help='correlations to measure'),
        ]

    def _run(self):
        checkpoint_dir = self.checkpoint_dir(self.save_dir)
        file_plot = '%s/plots/%s-%s/%s/%s_%s.pdf' % (checkpoint_dir, '%d', '%d', '%s', '%s', '%s')

        # figure out what each bench has
        data_sets = []
        architectures = []
        for bs in self.benchmark_sets:
            assert isinstance(bs, MiniNASTabularBenchmark)
            data_sets.append(set(bs.get_all_datasets()))
            architectures.append(set(bs.get_all_architecture_tuples()))

        # plot all set intersections
        for i0 in range(len(self.benchmark_sets)-1):
            for i1 in range(i0+1, len(self.benchmark_sets)):
                log_headline(self.logger, "correlating i0=%d and i1=%d" % (i0, i1), target_len=80)
                bench0, bench1 = self.benchmark_sets[i0], self.benchmark_sets[i1]

                self.logger.info("bench[%d]: %s" % (i0, bench0.get_name()))
                self.logger.info("bench[%d]: %s" % (i1, bench1.get_name()))

                # intersection of evaluated architectures
                arc0, arc1 = architectures[i0], architectures[i1]
                arc = list(arc0.intersection(arc1))
                self.logger.info("num architectures: num bench[%d] = %d, num bench[%d] = %d, num intersection = %d"
                                 % (i0, len(arc0), i1, len(arc1), len(arc)))
                if len(arc) == 0:
                    self.logger.info("skipping, can not correlate any architectures")
                    continue

                # intersection of evaluated data sets
                ds0, ds1 = data_sets[i0], data_sets[i1]
                ds, used_ds = list(ds0.intersection(ds1)), []
                if self.same_dataset:
                    used_ds = [(d, d) for d in ds]
                else:
                    for ds0_ in ds0:
                        for ds1_ in ds1:
                            used_ds.append((ds0_, ds1_))
                self.logger.info("data sets: bench[%d] = %s, bench[%d] = %s, intersection = %s, used combinations = %s"
                                 % (i0, ds0, i1, ds1, ds, used_ds))
                if len(used_ds) == 0:
                    self.logger.info("skipping, can not correlate any architectures")
                    continue

                # get all relevant results
                results0, results1 = [], []
                for arc_ in arc:
                    results0.append(bench0.get_by_arch_tuple(arc_))
                    results1.append(bench1.get_by_arch_tuple(arc_))

                # correlate
                for ds0_, ds1_ in used_ds:
                    for key in MiniResult.get_metric_keys():
                        name = 'all'
                        ds_str = ds0_ if ds0_ == ds1_ else "%s-%s" % (ds0_, ds1_)

                        type0 = self.benchmark_sets[i0].default_result_type
                        type1 = self.benchmark_sets[i1].default_result_type
                        values0 = [r.get(key, data_set=ds0_, result_type=type0) for r in results0]
                        values1 = [r.get(key, data_set=ds1_, result_type=type1) for r in results1]

                        values0 = np.nan_to_num(values0, nan=-1)
                        values1 = np.nan_to_num(values1, nan=-1)

                        self.correlation_cls[0].plot_correlations(
                            values0, values1, self.correlation_cls,
                            axes_names=('%s %s' % (bench0.get_name(), type0), '%s %s' % (bench1.get_name(), type1)),
                            show=False, save_path=file_plot % (i0, i1, name, key, ds_str))


@Register.task(search=True)
class EvalNetBenchTask(NetHPOTask):
    """
    Evaluate a trained super-network network on a bench
    """

    def __init__(self, args: Namespace, *args_, **kwargs):
        super().__init__(args, *args_, **kwargs)

        # restrictions
        assert len(self.objectives) == 1

        # bench part
        benchmark_set = self._parsed_meta_argument(Register.benchmark_sets, 'cls_benchmark', args, index=None)
        benchmark_set = benchmark_set.from_args(args, index=None)
        self.benchmark_set = SelfHPOUtils.bench_subspace(args, benchmark_set)
        assert isinstance(self.benchmark_set, MiniNASBenchmark)
        self.measure_top = self._parsed_argument('measure_top', self.args)
        # check if the cell architecture was shared during training
        self.num_normal = 1
        _, arc_shared = find_in_args(self.args, '.arc_shared')
        if not arc_shared:
            _, cell_order = find_in_args(self.args, '.cell_order')
            self.num_normal = cell_order.count('n')

        # nas metrics
        self.nas_cls = []
        for name in self._parsed_argument('nas_metrics', self.args, split_=True):
            self.nas_cls.append(Register.nas_metrics.get(name))

    @classmethod
    def meta_args_to_add(cls, algorithm=True) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        benchmark_sets = Register.benchmark_sets.filter_match_all(mini=True, tabular=True)
        return super().meta_args_to_add(algorithm=algorithm) + [
            MetaArgument('cls_benchmark', benchmark_sets, allowed_num=1, help_name='mini benchmark set to optimize on'),
        ]

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('measure_top', default=500, type=int, help='measure top-N bench architectures'),
            Argument('nas_metrics', default='ImprovementNasMetric, KendallTauNasMetric', type=str, help='metrics to calculate'),
        ]

    def _run(self, save=False):
        checkpoint_dir = self.checkpoint_dir(self.save_dir)

        # what are the best architectures in a surrogate benchmark...?
        if (self.measure_top > 0) and (not self.benchmark_set.is_tabular()):
            raise NotImplementedError("can not measure top-N networks on a non-tabular benchmark")

        # value space, already sorted by best
        sm = StrategyManager()
        svs = [v.arch_tuple for v in self.benchmark_set.get_all_sorted(['acc1'], [True])]
        arc_len = len(svs[0])
        if (self.num_normal > 1) and (len(svs)*self.num_normal == sm.get_num_choices(unique=True)):
            # compensate now for late architecture sharing by duplicating the indices
            svs = [tuple(list(v)*self.num_normal) for v in svs]
        self._svs = SpecificValueSpace(svs)

        # run
        algorithm, name_num = None, [(str(self.measure_top), self.measure_top), ('random', 9999999999)]
        for name, num in name_num:
            if algorithm is not None:
                algorithm.remove_saved_state()

            # tweak self params and let the super class run
            self._architecture_space = deepcopy(self._svs)
            self._architecture_space.specific_values = self._architecture_space.specific_values[:num]
            algorithm, population = super()._run(save=save)

            # compare, objective key
            obj_key = self.objectives[0].key
            net_values = []
            bench_values = defaultdict(list)
            for candidate in population.candidates:
                net_values.append(candidate.metrics.get(obj_key))
                r = self.benchmark_set.get_by_arch_tuple(candidate.values[:arc_len])
                for ds in r.get_data_sets():
                    bench_values[ds].append(r.get(kind=obj_key, data_set=ds))

            # plots, logging
            self.get_method().log_metrics({
                'net_bench/%s/num' % name: population.size
            })
            for ben_ds, ben_values in bench_values.items():
                for nas_cls in self.nas_cls:
                    # calculate metric
                    metric_dct = nas_cls.get_data(net_values, ben_values)

                    # plot
                    file_plot = '%s/plots/metrics/%s/%s/%s_%s.pdf' %\
                                (checkpoint_dir, name, ben_ds, obj_key, nas_cls.__name__)
                    nas_cls.plot(data=metric_dct, title='', legend=True, show=False, save_path=file_plot)

                    # log
                    for k, v in metric_dct.items():
                        self.get_method().log_metric_lists({
                            'net_bench/%s/%s/%s/%s/%s' % (name, ben_ds, obj_key, nas_cls.__name__, k): v
                        })


@Register.task(search=True)
class CreateSearchNetBenchTask(NetHPOTask):
    """
    Evaluate a s1 network and create a bench from the results
    this is an intermediate step to compare the prediction of several search network
    """

    def __init__(self, args: Namespace, *args_, **kwargs):
        super().__init__(args, *args_, **kwargs)

        for key in MiniResult.get_metric_keys():
            if not any([o.key == key for o in self.objectives + self.constraints]):
                self.logger.warning('Will not evaluate key "%s" on the network (not an objective/constraint)' % key)

        self.measure_min = self._parsed_argument('measure_min', args, index=None)
        benchmark_sets = self._parsed_meta_arguments(Register.benchmark_sets, 'cls_benchmarks', args, index=None)
        self.benchmark_sets = [bs.from_args(args, index=i) for i, bs in enumerate(benchmark_sets)]
        for bs in self.benchmark_sets:
            assert isinstance(bs, MiniNASTabularBenchmark)

    @classmethod
    def meta_args_to_add(cls, algorithm=True) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        benchmark_sets = Register.benchmark_sets.filter_match_all(mini=True, tabular=True)
        return super().meta_args_to_add(algorithm=algorithm) + [
            MetaArgument('cls_benchmarks', benchmark_sets,
                         help_name='optional benchmark sets, to evaluate specific architectures'),
        ]

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('measure_min', default=-1, type=int,
                     help='min amount of architectures to generate (but the hpo algorithm may evaluate less)'),
        ]

    def _run(self, save=False):
        # value spaces
        values = set()
        sm = StrategyManager()

        # add all evaluated architectures of the benchmarks
        for bs in self.benchmark_sets:
            assert isinstance(bs, MiniNASTabularBenchmark)
            l0, l1 = len(sm.ordered_names(unique=True)), bs.get_value_space().num_choices()
            assert l0 == l1, "Num choices of the network space (%d) and the bench space (%d) must match" % (l0, l1)
            for r in bs.get_all():
                values.add(r.arch_tuple)
        if len(values) > 0:
            self.logger.info("Added %d architectures from given benchmark set(s) to the list" % len(values))

        # if the space is smaller than desired, add random architectures
        network = self.get_method().get_network()
        assert isinstance(network, SearchUninasNetwork)
        net_space = sm.get_value_space()
        if self.measure_min > len(values):
            self.logger.info("Adding random architectures, have %d/%d" % (len(values), self.measure_min))
            while len(values) < self.measure_min:
                values.add(net_space.random_sample())

        # evaluate the given architectures
        self._architecture_space = SpecificValueSpace(list(values))
        algorithm, population = super()._run(save=save)

        # add info to the candidates, e.g. from profilers, such as loss/flops/latency/macs
        pass

        # create a new bench
        bench = MiniNASSearchTabularBenchmark.make_from_population(population, self.get_method())
        log_headline(self.logger, "Created bench file from super-network")
        bench.print_info(self.logger.info)
        bench.save_in_dir(self.save_dir)
        explore(bench, self.logger, n=10)
