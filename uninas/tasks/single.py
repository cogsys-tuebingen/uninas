from uninas.tasks.abstract import AbstractNetTask
from uninas.utils.args import MetaArgument, Argument, Namespace
from uninas.utils.loggers.python import log_headline
from uninas.register import Register


class SingleTask(AbstractNetTask):
    """
    A task that uses a single lightning module and a single trainer
    """

    def __init__(self, args: Namespace, *args_, **kwargs):
        super().__init__(args, *args_, **kwargs)

        # single method, single trainer
        log_headline(self.logger, 'setting up...')
        self.add_method()
        self.add_trainer(method=self.get_method(), save_dir=self.save_dir, num_devices=-1)
        self.log_detailed()

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

    def __init__(self, args: Namespace, *args_, **kwargs):
        super().__init__(args, *args_, **kwargs)
        cls_bss = self._parsed_meta_arguments(Register.benchmark_sets, 'cls_benchmark', args, index=None)
        self.benchmark_set = cls_bss[0].from_args(args, index=None) if len(cls_bss) > 0 else None

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + [
            MetaArgument('cls_benchmark', Register.benchmark_sets, allowed_num=(0, 1), use_index=False,
                         help_name='immediately look up the search result in this benchmark set (optional)'),
        ]

    def _run(self):
        """ execute the task """
        log_headline(self.logger, "Training")
        self.trainer[0].train_until_max_epoch()

        # try to immediately evaluate the result on the maybe-given bench db
        if self.benchmark_set is not None:
            gene = self.get_method().get_network().get_space_tuple(unique=True, flat=True)
            try:
                result = self.benchmark_set.get_by_arch_tuple(gene)
                assert result is not None, "Bench exists, but there if no result for gene %s" % str(gene)
                log_headline(self.logger, "Bench results")
                self.benchmark_set.print_info(self.logger.info)
                result.print(self.logger.info, prefix='\t')
                self.get_method().log_metrics(self.benchmark_set.get_result_dict(result))
            except Exception as e:
                self.logger.warning("Can not load a result from the bench db", exc_info=e)

        # save configs
        log_headline(self.logger, "Saving config(s)")
        self.get_method().save_configs("%s/network/" % self.checkpoint_dir(self.save_dir))


@Register.task()
class SingleRetrainTask(SingleTask):
    """
    A task that simply retrains a single network with a single trainer
    """

    def _run(self):
        """ execute the task """
        log_headline(self.logger, "Training")
        self.trainer[0].train_until_max_epoch()

        # save configs
        log_headline(self.logger, "Saving config")
        self.get_method().save_configs("%s/network/" % self.checkpoint_dir(self.save_dir))
