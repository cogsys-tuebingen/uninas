from uninas.tasks.abstract import AbstractTask
from uninas.methods.strategies.manager import StrategyManager
from uninas.methods.strategies.random import RandomChoiceStrategy
from uninas.optimization.common.profilers.abstract import AbstractProfiler
from uninas.utils.args import MetaArgument, Namespace
from uninas.utils.loggers.python import log_headline
from uninas.register import Register


@Register.task(search=True)
class SearchNetworkProfileTask(AbstractTask):
    """
    A task that profiles the modules of a single network.
    Requires the data set for shape information and batch size (the data itself can be fake).
    """

    def __init__(self, args: Namespace, wildcards: dict):
        AbstractTask.__init__(self, args, wildcards)

        # for architecture weights
        log_headline(self.logger, 'adding Strategy and Data')
        StrategyManager().add_strategy(RandomChoiceStrategy(max_epochs=1))

        # data
        data_set = self._parsed_meta_argument('cls_data', args, index=None)(args)
        self.batch_size = data_set.get_batch_size(train=False)

        # device handling
        self.devices_handler = self._parsed_meta_argument('cls_device', args, index=None)\
            .from_args(self.seed, args, index=None)
        self.mover = self.devices_handler.allocate_devices(num=-1)

        # network
        log_headline(self.logger, 'adding Network')
        self.net = self._parsed_meta_argument('cls_network', args, index=None).from_args(args)
        self.net.build(s_in=data_set.get_data_shape(), s_out=data_set.get_label_shape())
        self.net = self.mover.move_module(self.net)

        # profiler
        log_headline(self.logger, 'adding Profiler')
        self.profiler = self._parsed_meta_argument('cls_profiler', args, index=None)\
            .from_args(args, index=None, is_test_run=self.is_test_run)
        assert isinstance(self.profiler, AbstractProfiler)

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        networks = Register.networks.filter_match_all(search=True)
        return super().meta_args_to_add() + [
            MetaArgument('cls_device', Register.devices_managers, help_name='device manager', allowed_num=1),
            MetaArgument('cls_profiler', Register.profilers, help_name='profiler', allowed_num=1),
            MetaArgument('cls_data', Register.data_sets, help_name='data set', allowed_num=1),
            MetaArgument('cls_network', networks, help_name='network', allowed_num=1),
        ]

    def _profile_file(self, checkpoint_dir: str) -> str:
        return '%s/%s.%s.pt' % (checkpoint_dir, self.profiler.name, self.devices_handler.name)

    def _load(self, checkpoint_dir: str) -> bool:
        """ load """
        return self.profiler.load(self._profile_file(self.save_dir))

    def _run(self):
        """ execute the task """
        log_headline(self.logger, "Profiling")
        self.profiler.profile(self.net, self.mover, self.batch_size)
        self.profiler.save(self._profile_file(self.save_dir))
