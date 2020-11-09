import os
import shutil
import logging
import random
from typing import Union, List
import numpy as np
import torch
from uninas.utils.args import ArgsInterface, Argument, MetaArgument, Namespace, sanitize, save_as_json
from uninas.utils.misc import split
from uninas.utils.loggers.python import get_logger, log_headline, log_args
from uninas.utils.torch.misc import count_parameters
from uninas.utils.system import dump_system_info
from uninas.methods.abstract import AbstractMethod
from uninas.methods.strategies.manager import StrategyManager
from uninas.register import Register

cla_type = Union[str, List, None]


class AbstractTask(ArgsInterface):

    def __init__(self, args: Namespace, wildcards: dict):
        super().__init__()

        # args, seed
        self.args = args
        self.save_dir = self._parsed_argument('save_dir', args)
        self.is_test_run = self._parsed_argument('is_test_run', args)
        self.seed = self._parsed_argument('seed', args)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # maybe delete old dir, note arguments, save run_config
        if self._parsed_argument('save_del_old', args):
            shutil.rmtree(self.save_dir, ignore_errors=True)
        os.makedirs(self.save_dir, exist_ok=True)
        save_as_json(args, self.save_dir + 'task.run_config', wildcards)
        dump_system_info(self.save_dir + 'sysinfo.txt')

        # logging
        self.log_file = '%slog_task.txt' % self.save_dir
        self.logger = self.new_logger(None)
        log_args(self.logger, None, self.args, add_git_hash=True)
        Register.log_all(self.logger)

        # reset weight strategies so that consecutive tasks do not conflict with each other
        StrategyManager().reset()

        self.methods = []

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('is_test_run', default='False', type=str, help='test runs stop epochs early', is_bool=True),
            Argument('seed', default=0, type=int, help='random seed for the experiment'),
            Argument('note', default='note', type=str, help='just to take notes'),

            # saving
            Argument('save_dir', default='{path_tmp}', type=str, help='where to save', is_path=True),
            Argument('save_del_old', default='True', type=str, help='wipe the save dir before starting', is_bool=True),
        ]

    @classmethod
    def _add_meta_from_argsfile_to_args(cls, all_args: [str], meta_keys: [str], args_in_file: dict, overwrite=True):
        """ copy all meta arguments in 'meta_keys' and their respective arguments to the 'all_args' list """
        already_added = set()
        if not overwrite:
            for s in all_args:
                already_added.add(s.split('=')[0][2:])
        for key_meta in meta_keys:
            value_meta = args_in_file.get(key_meta)
            value_splits = split(sanitize(value_meta))
            for key_cls in value_splits:
                for k, v in args_in_file.items():
                    if k in already_added:
                        continue
                    if key_meta in k or key_cls in k:
                        all_args.append('--%s=%s' % (k, v))
                        already_added.add(k)
                        if key_meta == k:
                            print('\t\tusing "%s" as %s, copying arguments' % (v, key_meta))

    def get_first_method(self) -> AbstractMethod:
        raise NotImplementedError("This task may not use methods/networks at all")

    def checkpoint_dir(self, save_dir: str = None) -> str:
        return save_dir if save_dir is not None else self.save_dir

    def new_logger(self, index=None):
        return get_logger(name=index if index is None else str(index),
                          default_level=logging.DEBUG if self.is_test_run else logging.INFO,
                          save_file=self.log_file)

    def load(self, checkpoint_dir: str = None):
        """ load """
        log_headline(self.logger, 'Loading')
        checkpoint_dir = self.checkpoint_dir(checkpoint_dir)
        try:
            if not self._load(checkpoint_dir):
                self.logger.info('Did not load, maybe nothing to do: %s' % checkpoint_dir)
        except Exception as e:
            self.logger.error('Failed loading from checkpoint dir: "%s"' % checkpoint_dir, exc_info=e)
        return self

    def _load(self, checkpoint_dir: str) -> bool:
        """ load """
        return False

    def run(self):
        """ execute the task """
        self._run()
        for method in self.methods:
            method.flush_logging()
        self.logger.info("Done!")
        return self

    def _run(self):
        """ execute the task """
        raise NotImplementedError


class AbstractNetTask(AbstractTask):

    def __init__(self, args: Namespace, wildcards: dict):
        AbstractTask.__init__(self, args, wildcards)

        # device handling
        self.devices_handler = self._parsed_meta_argument('cls_device', args, None).from_args(self.seed, args, index=None)

        # classes
        self.cls_method = self._parsed_meta_argument('cls_method', args, None)
        self.cls_trainer = self._parsed_meta_argument('cls_trainer', args, None)

        # methods and trainers
        self.trainer = []

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        kwargs = Register.get_my_kwargs(cls)
        methods = Register.methods.filter_match_all(search=kwargs.get('search'))

        return super().meta_args_to_add() + [
            MetaArgument('cls_device', Register.devices_managers, help_name='device manager', allowed_num=1),
            MetaArgument('cls_trainer', Register.trainers, help_name='trainer', allowed_num=1),
            MetaArgument('cls_method', methods, help_name='method', allowed_num=1),
        ]

    def get_first_method(self) -> AbstractMethod:
        return self.methods[0]

    def add_method(self):
        """ adds a new method (lightning module) """
        # never try loading from checkpoint, since custom checkpoints are used
        # if checkpoint_file is not None and os.path.isfile(checkpoint_file):
        #     self.logger.info('Loading Lightning module from checkpoint "%s"' % checkpoint_file)
        #     return self.cls_method.load_from_checkpoint(checkpoint_file)
        method = self.cls_method(self.args)
        self.methods.append(method)

    def add_trainer(self, method: AbstractMethod, save_dir: str, num_devices=-1):
        """ adds a new trainer which saves to 'save_dir' and uses 'num_gpus' gpus """
        mover = self.devices_handler.allocate_devices(num_devices)
        logger = self.logger if self.devices_handler.get_num_free() == 0 else self.new_logger(len(self.trainer))
        trainer = self.cls_trainer(method=method,
                                   args=self.args,
                                   mover=mover,
                                   save_dir=save_dir,
                                   logger=logger,
                                   is_test_run=self.is_test_run)
        self.trainer.append(trainer)

    def log_methods_and_trainer(self):
        # log some things
        log_headline(self.logger, 'Trainer, Method, Data, ...')
        log_str = '{:<20}{}'
        for i, trainer in enumerate(self.trainer):
            self.logger.info(log_str.format('Trainer', trainer.str()))
            if hasattr(trainer, 'optimizers'):
                for j, optimizer in enumerate(trainer.optimizers):
                    self.logger.info(log_str.format('Optimizer (%d)' % j, str(optimizer)))
                for j, scheduler in enumerate(trainer.schedulers):
                    self.logger.info(log_str.format('Scheduler (%d)' % j, scheduler.str()))
        for j, method in enumerate(self.methods):
            self.logger.info(log_str.format('Method', method.str()))
            self.logger.info(log_str.format('Data set', method.data_set.str()))
            self.logger.info(log_str.format(' > train/eval', method.data_set.list_train_transforms()))
            self.logger.info(log_str.format(' > test', method.data_set.list_test_transforms()))
            self.logger.info(log_str.format('Criterion', str(method.criterion)))
            for i, m in enumerate(method.metrics):
                self.logger.info(log_str.format('Metric (%d)' % i, m.str()))
            for i, r in enumerate(method.regularizers):
                self.logger.info(log_str.format('Regularizer (%d)' % i, r.str()))

            strategies = StrategyManager().get_strategies_list()
            if len(strategies) > 0:
                if len(self.methods) > 1:
                    self.logger.info('Weight strategies:')
                else:
                    log_headline(self.logger, 'Weight strategies')
                for strategy in strategies:
                    self.logger.info('%s' % strategy.str())
                    for r in strategy.get_requested_weights():
                        self.logger.info(
                            '\t{:<30} {:>2} choices, used {}x'.format(r.name, r.num_choices(), r.num_requests()))
                self.logger.info('All weights in request order (not unique):')
                self.logger.info('\t%s', str(StrategyManager().ordered_names(unique=False)))
                self.logger.info('All weights in request order (unique):')
                self.logger.info('\t%s', str(StrategyManager().ordered_names(unique=True)))

            if len(self.methods) > 1:
                self.logger.info('Model:')
            else:
                log_headline(self.logger, 'Model')
            self.logger.info(method.get_network().str())
            self.logger.info("Param count: %d", count_parameters(method.get_network()))

    def _run(self):
        """ execute the task """
        raise NotImplementedError
