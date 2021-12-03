import os
import shutil
import logging
import random
from typing import Union, List
import numpy as np
import torch
from uninas.utils.args import ArgsInterface, Argument, MetaArgument, Namespace, sanitize, save_as_json
from uninas.utils.misc import split
from uninas.utils.loggers.python import LoggerManager, log_headline, log_in_columns, log_args
from uninas.utils.paths import get_task_config_path
from uninas.utils.system import dump_system_info
from uninas.methods.abstract_method import AbstractMethod
from uninas.methods.strategy_manager import StrategyManager
from uninas.register import Register

cla_type = Union[str, List, None]


class AbstractTask(ArgsInterface):

    def __init__(self, args: Namespace, wildcards: dict, descriptions: dict = None):
        super().__init__()

        # args, seed
        self.args = args
        self.save_dir = self._parsed_argument('save_dir', args)
        self.is_test_run = self._parsed_argument('is_test_run', args)
        self.seed = self._parsed_argument('seed', args)
        self.is_deterministic = self._parsed_argument('is_deterministic', args)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.is_deterministic:
            # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.set_deterministic(self.is_deterministic)

        # maybe delete old dir, note arguments, save run_config
        if self._parsed_argument('save_del_old', args):
            shutil.rmtree(self.save_dir, ignore_errors=True)
        os.makedirs(self.save_dir, exist_ok=True)
        save_as_json(args, get_task_config_path(self.save_dir), wildcards)
        dump_system_info(self.save_dir + 'sysinfo.txt')

        # logging
        self.log_file = '%slog_task.txt' % self.save_dir
        LoggerManager().set_logging(default_save_file=self.log_file)
        self.logger = self.new_logger(index=None)
        log_args(self.logger, None, self.args, add_git_hash=True, descriptions=descriptions)
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
            Argument('is_deterministic', default='False', type=str, help='use deterministic operations', is_bool=True),
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

    def get_method(self) -> AbstractMethod:
        """ get the only existing method """
        assert len(self.methods) == 1, "Must have exactly one method, but %d exist" % len(self.methods)
        return self.methods[0]

    def checkpoint_dir(self, save_dir: str = None) -> str:
        return save_dir if save_dir is not None else self.save_dir

    def new_logger(self, index: int = None):
        return LoggerManager().get_logger(
            name=index if index is None else str(index),
            default_level=logging.DEBUG if self.is_test_run else logging.INFO,
            save_file=self.log_file)

    def load(self, checkpoint_dir: str = None) -> 'AbstractTask':
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

    def run(self) -> 'AbstractTask':
        """ execute the task """
        try:
            self._run()
            for method in self.methods:
                method.flush_logging()
            self.logger.info("Done!")
            return self
        except Exception as e:
            raise e
        finally:
            LoggerManager().cleanup()

    def _run(self):
        """ execute the task """
        raise NotImplementedError


class AbstractNetTask(AbstractTask):

    def __init__(self, args: Namespace, *args_, **kwargs):
        AbstractTask.__init__(self, args, *args_, **kwargs)

        # device handling
        cls_dev_handler = self._parsed_meta_argument(Register.devices_managers, 'cls_device', args, None)
        self.devices_handler = cls_dev_handler.from_args(self.seed, self.is_deterministic, args, index=None)

        # classes
        self.cls_method = self._parsed_meta_argument(Register.methods, 'cls_method', args, None)
        self.cls_trainer = self._parsed_meta_argument(Register.trainers, 'cls_trainer', args, None)

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

    def log_detailed(self):
        # log some things
        log_headline(self.logger, 'Trainer, Method, Data, ...')
        rows = [('Trainer', '')]
        for i, trainer in enumerate(self.trainer):
            rows.append((' (%d)' % i, trainer.str()))
        log_in_columns(self.logger, rows)

        for i, method in enumerate(self.methods):
            log_headline(self.logger, "Method %d/%d" % (i+1, len(self.methods)), target_len=80)
            method.log_detailed(self.logger)

        StrategyManager().log_detailed(self.logger)

    def _run(self):
        """ execute the task """
        raise NotImplementedError
