import os
import shutil
from uninas.methods.abstract_method import AbstractMethod
from uninas.methods.strategy_manager import StrategyManager
from uninas.optimization.benchmarks.mini.tabular_search import MiniNASTabularBenchmark, MiniNASSearchTabularBenchmark
from uninas.optimization.hpo.uninas.algorithms.randomly import RandomlyEval
from uninas.optimization.estimators.net import AbstractNetEstimator
from uninas.training.trainer.abstract import AbstractTrainerFunctions
from uninas.training.trainer.simple import SimpleTrainer
from uninas.training.callbacks.abstract import AbstractCallback
from uninas.training.callbacks.checkpoint import CheckpointCallback
from uninas.utils.loggers.python import LoggerManager, log_headline
from uninas.utils.args import MetaArgument, Argument, Namespace
from uninas.utils.torch.misc import reset_bn as reset_bn_fun
from uninas.register import Register


@Register.training_callback()
class CreateBenchCallback(AbstractCallback):
    """
    Periodically create a bench from the currently trained super-network
    - requires a search network
    - requires at least one objective to evaluate
    """

    def __init__(self, save_dir: str, index: int, args: Namespace,
                 each_epochs: int, reset_bn: bool, benchmark_path: str):
        """
        """
        super().__init__(save_dir, index=index)
        self.logger = LoggerManager().get_logger()
        self.objectives = []
        self._args = args
        self.each_epochs = each_epochs
        self.reset_bn = reset_bn
        self.benchmark = MiniNASTabularBenchmark.load(benchmark_path)
        assert self.benchmark.is_tabular(), "Can only fully evaluate a tabular benchmark set"
        self.benchmark_space = self.benchmark.get_specific_value_space()

        # paths
        _save_dir = "%s/callbacks/%s/%d/" % (self._save_dir, self.__class__.__name__, self._index)
        self._path_net_cur = "%s/weights/cur.pt" % _save_dir
        self._path_net_obj = "%s/weights/obj.pt" % _save_dir if self.reset_bn else self._path_net_cur
        self._save_path_benches = "%s/benches/epoch_%s.pt" % (_save_dir, '%d')

    @classmethod
    def from_args(cls, save_dir: str, args: Namespace, index: int) -> 'AbstractCallback':
        parsed = cls._all_parsed_arguments(args, index=index)
        return cls(save_dir, index, args=args, **parsed)

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + [
            MetaArgument('cls_cb_objectives', Register.hpo_estimators, allowed_num=(1, -1),
                         help_name='objectives (no constraints) to estimate'),
        ]

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument("each_epochs", default=1, type=int, help="evaluate the architectures each n epochs"),
            Argument('reset_bn', default='False', type=str, help='reset batch norm stats for evaluation', is_bool=True),
            Argument("benchmark_path", default="{path_data}/bench/nats/nats_bench_1.1_mini.pt", type=str,
                     help="evaluate all architectures in this benchmark"),
        ]

    def setup(self, trainer: AbstractTrainerFunctions, pl_module: AbstractMethod, stage: str):
        """ Called when fit or test begins """
        # assert some things
        assert pl_module.matches_registered_properties(search=True, single_path=True),\
            "Must use a single-path search network"
        n0, n1 = len(StrategyManager().get_num_choices(unique=True)), self.benchmark.get_value_space().num_choices()
        assert n0 == n1, "Number of network choice (%d) does not match bench choices (%d)" % (n0, n1)
        assert isinstance(trainer, SimpleTrainer)

        # set up objectives
        est = self._parsed_meta_arguments(Register.hpo_estimators, 'cls_cb_objectives', self._args, index=self._index)
        for i, e in enumerate(est):
            assert issubclass(e, AbstractNetEstimator)
            estimator = e(args=self._args, index=i, trainer=trainer, method=pl_module, load_path=self._path_net_obj)
            assert estimator.is_objective() and not estimator.is_constraint(), "Can only use objectives"
            self.objectives.append(estimator)
        assert len(self.objectives) > 0, "Pointless callback if there are no objectives"

    def on_train_epoch_end(self, trainer: AbstractTrainerFunctions,
                           pl_module: AbstractMethod,
                           log_dict: dict = None):
        """ Called when the train epoch ends. """
        if (pl_module.current_epoch + 1) % self.each_epochs != 0:
            return

        num_eval = 5 if trainer.is_test_run() else len(self.benchmark_space)
        fmt = "evaluating {num} benchmark architectures{x}"
        log_headline(self.logger, fmt.format(
            num=num_eval,
            x=" (of %d, this is a test run)" % len(self.benchmark_space) if trainer.is_test_run() else ""))

        # save current network weights
        CheckpointCallback.save(self._path_net_cur, pl_module)
        pl_module.get_network().set_forward_strategy(False)

        # maybe reset bn stats, save current network weights
        if self.reset_bn:
            reset_bn_fun(pl_module.get_network())
            CheckpointCallback.save(self._path_net_obj, pl_module)

        # run eval over entire bench, make a bench from that, save it
        eval_ = RandomlyEval(value_space=self.benchmark_space, logger=self.logger, save_file=None,
                             objectives=self.objectives, num_eval=num_eval)
        eval_.search()
        population = eval_.get_total_population(sort=False)
        bench = MiniNASSearchTabularBenchmark.make_from_population(population, pl_module)
        bench.save(self._save_path_benches % pl_module.current_epoch)

        # recover current network weights
        CheckpointCallback.load(self._path_net_cur, pl_module=pl_module)
        pl_module.get_network().set_forward_strategy(True)
        log_headline(self.logger, "benchmark architectures evaluated")

    def teardown(self, trainer: AbstractTrainerFunctions, pl_module: AbstractMethod, stage: str):
        """ Called when fit or test ends """
        shutil.rmtree(os.path.dirname(self._path_net_cur))
