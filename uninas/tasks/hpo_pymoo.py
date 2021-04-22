import os
import shutil
import numpy as np
from uninas.tasks.abstract import AbstractTask, AbstractNetTask
from uninas.optimization.task import common_s2_net_args_to_add, common_s2_extend_args, common_s2_prepare_run
from uninas.optimization.hpo.pymoo.algorithms.abstract import AbstractPymooAlgorithm, Algorithm
from uninas.optimization.hpo.pymoo.problem import PymooProblem, BenchPymooProblem
from uninas.optimization.hpo.pymoo.terminations import AbstractPymooTermination, Termination
from uninas.optimization.hpo.pymoo.result import PymooResultWrapper
from uninas.optimization.estimators.abstract import AbstractEstimator
from uninas.utils.paths import replace_standard_paths
from uninas.utils.args import MetaArgument, Argument, Namespace
from uninas.utils.loggers.python import log_headline, Logger
from uninas.register import Register
from uninas.builder import Builder


class PymooHPOUtils:
    """
    shared parts
    """

    @staticmethod
    def prepare(cls: AbstractTask.__class__, logger: Logger, estimator_kwargs: dict, args: Namespace, index=None) \
            -> (Algorithm, [], Termination):
        """
        :param cls:
        :param logger:
        :param estimator_kwargs:
        :param args: global namespace
        :param index: index of the task
        :return: algorithm class, estimators, termination
        """

        # pymoo hpo algorithm
        cls_algorithm = cls._parsed_meta_argument(Register.hpo_pymoo_algorithms, 'cls_hpo_pymoo_algorithm', args, index=index)
        assert issubclass(cls_algorithm, AbstractPymooAlgorithm), 'Method must have class methods to optimize the arc'
        algorithm = cls_algorithm.from_args(args)

        # estimators
        log_headline(logger, 'adding network estimators')
        estimators = []
        for i, e in enumerate(cls._parsed_meta_arguments(Register.hpo_estimators, 'cls_hpo_estimators', args, index=index)):
            estimator = e(args=args, index=i, **estimator_kwargs)
            estimators.append(estimator)
            logger.info(estimator.str())

        # termination
        log_headline(logger, 'adding algorithm termination')
        cls_terminator = cls._parsed_meta_argument(Register.hpo_pymoo_terminators, 'cls_hpo_pymoo_termination', args, index=index)
        assert issubclass(cls_terminator, AbstractPymooTermination),\
            "termination must be a subclass of %s" % AbstractPymooTermination.__name__
        termination = cls_terminator.from_args(args=args, index=None)
        logger.info(cls_terminator().str())

        return algorithm, estimators, termination

    @classmethod
    def meta_args_to_add(cls, estimator_filter: dict = None) -> [MetaArgument]:
        estimators = Register.hpo_estimators
        if isinstance(estimator_filter, dict):
            estimators = estimators.filter_match_all(**estimator_filter)
        return [
            MetaArgument('cls_hpo_estimators', estimators, help_name='performance estimators', allowed_num=(1, -1), allow_duplicates=True),
            MetaArgument('cls_hpo_pymoo_algorithm', Register.hpo_pymoo_algorithms, help_name='hyper-parameter algorithm', allowed_num=1),
            MetaArgument('cls_hpo_pymoo_termination', Register.hpo_pymoo_terminators, help_name='algorithm termination criteria', allowed_num=1),
        ]

    @staticmethod
    def run(problem: PymooProblem, algorithm: Algorithm, termination: Termination, seed: int,
            logger: Logger, checkpoint_dir: str) -> PymooResultWrapper:
        # run
        log_headline(logger, 'running')
        wrapper = PymooResultWrapper.minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            pf=problem.pareto_front(use_cache=False),
            save_history=True,
            verbose=True
        )

        # log, plot
        wrapper.log_best(logger)
        wrapper.plot_all_f(checkpoint_dir)
        wrapper.plot_hv(checkpoint_dir)

        return wrapper


@Register.task(search=True)
class MiniBenchPymooHPOTask(AbstractTask):
    """
    A hyper-parameter optimization task without networks/methods, purely on a given mini-bench
    """

    def __init__(self, args: Namespace, wildcards: dict, *args_, **kwargs):
        AbstractTask.__init__(self, args, wildcards, *args_, **kwargs)
        benchmark_set = self._parsed_meta_argument(Register.benchmark_sets, 'cls_benchmark', args, index=None)
        self.benchmark_set = benchmark_set.from_args(args, index=None)
        self.plot_true_pareto = self._parsed_argument('plot_true_pareto', args)

        estimator_kwargs = dict(mini_api=self.benchmark_set)
        self.algorithm, self.estimators, self.termination = PymooHPOUtils.prepare(self, self.logger, estimator_kwargs, args)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return AbstractTask.args_to_add(index) + [
            Argument('plot_true_pareto', default='False', type=str, help='add the true pareto front', is_bool=True),
        ]

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        benchmark_sets = Register.benchmark_sets.filter_match_all(mini=True, tabular=True)
        return super().meta_args_to_add() + [
            MetaArgument('cls_benchmark', benchmark_sets, allowed_num=1, help_name='mini benchmark set to optimize on'),
        ] + PymooHPOUtils.meta_args_to_add(estimator_filter=dict(requires_bench=True))

    def get_estimator_kwargs(self) -> dict:
        """ kwargs for each estimator """
        return dict(mini_api=self.benchmark_set)

    def get_problem(self, estimators: [AbstractEstimator]) -> PymooProblem:
        """ define the problem """
        return BenchPymooProblem(estimators, self.benchmark_set, calc_pareto=self.plot_true_pareto)

    def run(self):
        return PymooHPOUtils.run(self.get_problem(self.estimators), self.algorithm, self.termination, self.seed,
                                 self.logger, self.checkpoint_dir(self.save_dir))


@Register.task(search=True)
class NetPymooHPOTask(AbstractNetTask):
    """
    An s2 task (trying to figure out the optimal network architecture of a trained s1 network)
    the chosen algorithm contains the exact optimization approach
    """

    def __init__(self, args: Namespace, *args_, **kwargs):
        super().__init__(args, *args_, **kwargs)

        # args
        self.s1_path = replace_standard_paths(self._parsed_argument('s1_path', args))
        self.reset_bn = self._parsed_argument('reset_bn', args)

        # files
        self.tmp_load_path = '%s/checkpoints/checkpoint.tmp.pt' % self.save_dir
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
        self.algorithm, self.estimators, self.termination = PymooHPOUtils.prepare(self, self.logger, estimator_kwargs, args)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + common_s2_net_args_to_add()

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + PymooHPOUtils.meta_args_to_add(estimator_filter=dict(requires_bench=False))

    @classmethod
    def extend_args(cls, args_list: [str]):
        """
        allow modifying the arguments list before other classes' arguments are dynamically added
        this should be used sparsely, as it is hard to keep track of
        """
        common_s2_extend_args(cls, args_list)

    def run(self):
        common_s2_prepare_run(self.logger, self.trainer, self.s1_path, self.tmp_load_path, self.reset_bn, self.methods)
        checkpoint_dir = self.checkpoint_dir(self.save_dir)
        candidate_dir = '%s/candidates/' % checkpoint_dir

        # get problem, run
        xu = np.array([n - 1 for n in self.get_method().strategy_manager.get_num_choices(unique=True)])
        xl = np.zeros_like(xu)
        problem = PymooProblem(estimators=self.estimators, xl=xl, xu=xu, n_var=len(xu))
        wrapper = PymooHPOUtils.run(problem, self.algorithm, self.termination, self.seed, self.logger, checkpoint_dir)

        # save results
        for sr in wrapper.sorted_best():
            self.get_method().get_network().forward_strategy(fixed_arc=tuple(sr.x))
            Builder.save_config(self.get_method().get_network().config(finalize=True),
                                candidate_dir, 'candidate-%s' % '-'.join([str(xs) for xs in sr.x]))
