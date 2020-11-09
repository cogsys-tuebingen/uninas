from uninas.tasks.abstract import AbstractNetTask
from uninas.utils.args import Argument, Namespace
from uninas.utils.loggers.python import log_headline
from uninas.benchmarks.mini import MiniNASBenchApi
from uninas.builder import Builder
from uninas.register import Register


class SingleTask(AbstractNetTask):
    """
    A task that uses a single lightning module and a single trainer
    """

    def __init__(self, args: Namespace, wildcards: dict):
        super().__init__(args, wildcards)

        # single method, single trainer
        log_headline(self.logger, 'adding Method, Trainer, ...')
        self.add_method()
        self.add_trainer(method=self.methods[0], save_dir=self.save_dir, num_devices=-1)
        self.log_methods_and_trainer()

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + []

    def _load(self, checkpoint_dir: str) -> bool:
        """ load """
        return self.trainer[0].load(checkpoint_dir)

    def _run(self):
        """ execute the task """
        raise NotImplementedError


@Register.task(search=True)
class SingleSearchTask(SingleTask):
    """
    A task that searches a single network with a single trainer, using a method.
    If a mini-bench is given, attempts to immediately find the search result in there.
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('mini_bench_path', default='', type=str, help='evaluate the result on a mini bench', is_path=True),
        ]

    def _run(self):
        """ execute the task """
        log_headline(self.logger, "Training")
        self.trainer[0].train_until_max_epoch()

        # try to immediately evaluate the result on the maybe-given bench db
        mini_bench_path = self._parsed_argument('mini_bench_path', self.args)
        if len(mini_bench_path) > 0:
            try:
                mini_bench = MiniNASBenchApi.load(mini_bench_path)
                gene = self.methods[0].get_network().get_space_tuple(unique=True, flatten=True)
                result = mini_bench.get_by_arch_tuple(gene)
                assert result is not None, "Bench exists, but there if no result for gene %s" % str(gene)
                log_headline(self.logger, "Bench results")
                mini_bench.print_info(self.logger.info)
                result.print(self.logger.info, prefix='\t')
                mini_bench.log_result(result, self.methods[0].logger)
            except Exception as e:
                self.logger.warning("Can not load a result from the bench db at '%s'" % mini_bench_path, exc_info=e)

        # save configs
        log_headline(self.logger, "Saving config(s)")
        self.methods[0].save_configs("%s/network/" % self.checkpoint_dir(self.save_dir))


@Register.task()
class SingleRetrainTask(SingleTask):
    """
    A task that simply retrains a single network with a single trainer
    forcibly uses RetrainMethod and adds the argument required for the method to properly load the desired network

    if config_only is used, it is not necessary to add any network-related arguments, as they will be ignored
    otherwise only the cells of the given config will be taken, head/stem or new cells can be added
    """

    def _run(self):
        """ execute the task """
        log_headline(self.logger, "Training")
        self.trainer[0].train_until_max_epoch()

        # save configs
        log_headline(self.logger, "Saving config")
        self.methods[0].save_configs("%s/network/" % self.checkpoint_dir(self.save_dir))
