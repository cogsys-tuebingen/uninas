import os
import shutil
import numpy as np
from uninas.tasks.abstract import AbstractTask, AbstractNetTask
from uninas.optimization.common.task import common_s2_net_args_to_add, common_s2_extend_args, common_s2_prepare_run
from uninas.optimization.hpo_pymoo.algorithms.abstract import AbstractPymooAlgorithm, Algorithm
from uninas.optimization.common.estimators.abstract import AbstractEstimator
from uninas.optimization.hpo_pymoo.problem import PymooProblem, BenchPymooProblem
from uninas.optimization.hpo_pymoo.terminations import AbstractPymooTermination, Termination
from uninas.optimization.hpo_pymoo.result import PymooResultWrapper
from uninas.utils.paths import replace_standard_paths
from uninas.utils.args import MetaArgument, Argument, Namespace
from uninas.utils.loggers.python import log_headline, Logger
from uninas.benchmarks.mini import MiniNASBenchApi
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
        cls_algorithm = cls._parsed_meta_argument('cls_hpo_pymoo_algorithm', args, index=index)
        assert issubclass(cls_algorithm, AbstractPymooAlgorithm), 'Method must have class methods to optimize the arc'
        algorithm = cls_algorithm.from_args(args)

        # estimators
        log_headline(logger, 'adding network estimators')
        estimators = []
        for i, e in enumerate(cls._parsed_meta_arguments('cls_hpo_estimators', args, index=index)):
            estimator = e(args=args, index=i, **estimator_kwargs)
            estimators.append(estimator)
            logger.info(estimator.str())

        # termination
        log_headline(logger, 'adding algorithm termination')
        cls_terminator = cls._parsed_meta_argument('cls_hpo_pymoo_termination', args, index=index)
        assert issubclass(cls_terminator, AbstractPymooTermination),\
            "termination must be a subclass of %s" % AbstractPymooTermination.__name__
        termination = cls_terminator.from_args(args=args, index=None)
        logger.info(cls_terminator().str())

        return algorithm, estimators, termination

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        return [
            MetaArgument('cls_hpo_estimators', Register.hpo_estimators, help_name='performance estimators', allowed_num=(1, -1)),
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

    def __init__(self, args: Namespace, wildcards: dict):
        AbstractTask.__init__(self, args, wildcards)
        self.mini_bench = MiniNASBenchApi.load(self._parsed_argument('mini_bench_path', args))
        self.mini_bench_dataset = self._parsed_argument('mini_bench_dataset', args)
        self.plot_true_pareto = self._parsed_argument('plot_true_pareto', args)

        estimator_kwargs = dict(mini_api=self.mini_bench, mini_api_set=self.mini_bench_dataset)
        self.algorithm, self.estimators, self.termination = PymooHPOUtils.prepare(self, self.logger, estimator_kwargs, args)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return AbstractTask.args_to_add(index) + [
            Argument('mini_bench_path', default='{path_data}/mini.pt', type=str, help='', is_path=True),
            Argument('mini_bench_dataset', default='cifar10', type=str, help=''),
            Argument('plot_true_pareto', default='False', type=str, help='add the true pareto front', is_bool=True),
        ]

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + PymooHPOUtils.meta_args_to_add()

    def get_estimator_kwargs(self) -> dict:
        """ kwargs for each estimator """
        return dict(mini_api=self.mini_bench, mini_api_set=self.mini_bench_dataset)

    def get_problem(self, estimators: [AbstractEstimator]) -> PymooProblem:
        """ define the problem """
        return BenchPymooProblem(estimators, self.mini_bench, calc_pareto=self.plot_true_pareto)

    def run(self):
        return PymooHPOUtils.run(self.get_problem(self.estimators), self.algorithm, self.termination, self.seed,
                                 self.logger, self.checkpoint_dir(self.save_dir))


@Register.task(search=True)
class NetPymooHPOTask(AbstractNetTask, PymooHPOUtils):
    """
    An s2 task (trying to figure out the optimal network architecture of a trained s1 network)
    the chosen algorithm contains the exact optimization approach
    """

    def __init__(self, args: Namespace, wildcards: dict):
        AbstractNetTask.__init__(self, args, wildcards)

        # args
        self.s1_path = replace_standard_paths(self._parsed_argument('s1_path', args))
        self.reset_bn = self._parsed_argument('reset_bn', args)

        # files
        self.tmp_load_path = '%s/checkpoints/checkpoint.tmp.pt' % self.save_dir
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
        return super().meta_args_to_add() + PymooHPOUtils.meta_args_to_add()

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
        file_candidate = '%s/candidates/%s.network_config' % (checkpoint_dir, 'candidate-%s')

        # get problem, run
        xu = np.array([n-1 for n in self.methods[0].strategy.ordered_num_choices(unique=True)])
        xl = np.zeros_like(xu)
        problem = PymooProblem(estimators=self.estimators, xl=xl, xu=xu, n_var=len(xu))
        wrapper = PymooHPOUtils.run(problem, self.algorithm, self.termination, self.seed, self.logger, checkpoint_dir)

        # save results
        for sr in wrapper.sorted_best():
            self.methods[0].get_network().forward_strategy(fixed_arc=tuple(sr.x))
            Builder.save_config(self.methods[0].get_network().config(finalize=True),
                                file_candidate % '-'.join([str(xs) for xs in sr.x]))
