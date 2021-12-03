import logging
import os
from torch.optim.optimizer import Optimizer
from uninas.methods.abstract_method import AbstractMethod
from uninas.training.trainer.abstract import AbstractTrainerFunctions
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.training.callbacks.checkpoint import CheckpointCallback
from uninas.training.clones.abstract import AbstractMethodClone
from uninas.training.schedulers.abstract import AbstractScheduler
from uninas.utils.loggers.resources import ResourceLogThread
from uninas.utils.loggers.python import LoggerManager
from uninas.utils.loggers.exp import AbstractExpLogger
from uninas.utils.args import ArgsInterface, MetaArgument, Argument, Namespace
from uninas.register import Register


class AbstractTrainer(ArgsInterface, AbstractTrainerFunctions):
    num_test_steps = 10
    can_use_ema = True
    can_eval_n = True
    can_step_opt_n = True

    def __init__(self, method: AbstractMethod, args: Namespace, save_dir: str, mover: AbstractDeviceMover, logger=None,
                 is_test_run=False):
        """

        :param method:
        :param args: global argparse namespace
        :param save_dir:
        :param logger:
        :param mover: object that wraps moving modules and tensors to devices
        :param is_test_run: test runs stop quickly
        """
        super().__init__()
        # args
        self.args = args
        self.max_epoch, self.stop_epoch = self._parsed_arguments(['max_epochs', 'stop_epoch'], args)
        log_fs, log_ram, log_device = self._parsed_arguments(['log_fs', 'log_ram', 'log_device'], args)
        self._is_test_run = is_test_run

        # method
        self.method = method

        # method clones, not yet initialized
        self.method_clones = []
        for i, clone_cls in enumerate(self._parsed_meta_arguments(Register.training_clones, 'cls_clones', self.args)):
            self.method_clones.append(clone_cls.from_args(self.args, index=i))

        # dirs, files
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        # logging basic
        self.logger = self.get_logger(logger, self._is_test_run, self.save_dir)
        logger_save_dir = '%sexp/' % save_dir
        os.makedirs(logger_save_dir, exist_ok=True)

        # device
        self.mover = mover
        self.mover.empty_cache()
        self.logger.info("Using device: %s" % self.mover.name)

        # log resources
        td = 5 if self._is_test_run else 300
        exp_logger = AbstractExpLogger.collection(logger_save_dir, args, self._parsed_meta_arguments(Register.exp_loggers, 'cls_exp_loggers', args, index=None))
        self.resource_logger = ResourceLogThread(exp_logger=exp_logger, seconds=td,
                                                 mover=self.mover if log_device else None,
                                                 log_fs=log_fs, log_ram=log_ram)
        self.resource_logger.start()
        self.logger.info("Continuously logging (devices=%s, RAM=%s, file_system=%s) each %ds" %
                         (self.mover.name, str(log_ram), str(log_fs), td))

        # log experiment data to e.g. tensorboard
        self.exp_logger = AbstractExpLogger.collection(logger_save_dir, args, self._parsed_meta_arguments(Register.exp_loggers, 'cls_exp_loggers', args, index=None))

        # eval/test the last n steps
        self.eval_last, self.test_last = self._parsed_arguments(['eval_last', 'test_last'], args, index=None)
        if not self.can_eval_n:
            self.logger.info('This trainer can not eval/test the last n steps, the arguments are just for consistency.')
        if not self.can_step_opt_n:
            self.logger.info('This trainer can not step the optimizer(s) each n steps, but will always step per epoch')

        # callbacks
        self.callbacks = [cls.from_args(self.save_dir, self.args, index=i)
                          for i, cls in enumerate(self._parsed_meta_arguments(Register.training_callbacks, 'cls_callbacks', args, index=None))]

        # gradient accumulation
        self.accumulate_batches = self._parsed_argument('accumulate_batches', args, index=None)
        self._acc_step = 0

    def _trigger_callbacks(self, callback_fun: str, *args, **kwargs):
        for c in self.callbacks:
            getattr(c, callback_fun)(*args, **kwargs)

    def get_save_dir(self) -> str:
        return self.save_dir

    def is_test_run(self) -> bool:
        return self._is_test_run

    def _str_dict(self) -> dict:
        dct = super()._str_dict()
        dct.update(dict(max_epoch=self.max_epoch, eval_last=self.eval_last, test_last=self.test_last))
        return dct

    def get_method(self) -> AbstractMethod:
        assert isinstance(self.method, AbstractMethod)
        return self.method

    def get_method_clones(self) -> [AbstractMethodClone]:
        """ get the method clones """
        return self.method_clones

    @classmethod
    def checkpoint_file(cls, save_dir: str, name='checkpoint.pt') -> str:
        return '%s%s' % (save_dir, name)

    @classmethod
    def get_logger(cls, logger=None, is_test_run=False, save_dir='/tmp/', suffix=''):
        """ new logger if required """
        if logger is not None:
            return logger
        return LoggerManager().get_logger(default_level=logging.DEBUG if is_test_run else logging.INFO,
                                          save_file='%slog_trainer%s.txt' % (save_dir, suffix))

    @classmethod
    def num_opt_steps(cls, loader, num_gpus=1, is_test_run=False) -> float:
        """ schedulers that step each n steps (each x.y epochs) should still step each x.y epochs when testing """
        if is_test_run:
            return num_gpus * loader.batch_size * len(loader) / cls.num_test_steps
        return num_gpus * loader.batch_size

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('max_epochs', default=1, type=int, help='max training epochs, affects schedulers + regularizers'),
            Argument('stop_epoch', default=-1, type=int, help='stop after training n epochs anyway, if > 0'),

            Argument('log_fs', default='True', type=str, help='log file system usage', is_bool=True),
            Argument('log_ram', default='True', type=str, help='log RAM usage', is_bool=True),
            Argument('log_device', default='True', type=str, help='log device usage', is_bool=True),

            Argument('eval_last', default=10, type=int, help='run eval for the last n epochs, always if <0'),
            Argument('test_last', default=10, type=int, help='run test for the last n epochs, always if <0'),

            Argument('accumulate_batches', default=1, type=int,
                     help='accumulate gradients over n batches before stepping updating. '
                          'Does not change the learning rate, '
                          'may cause issues when there are multiple alternating optimizers'),
        ]

    @classmethod
    def meta_args_to_add(cls, has_log_dict=True) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        callbacks = Register.training_callbacks
        if not has_log_dict:
            callbacks = callbacks.filter_match_all(requires_log_dict=False)
        return super().meta_args_to_add() + [
            MetaArgument('cls_callbacks', callbacks, help_name='training callbacks', allow_duplicates=True),
            MetaArgument('cls_clones', Register.training_clones, help_name='training clones', allow_duplicates=True),
            MetaArgument('cls_exp_loggers', Register.exp_loggers, help_name='experiment logger', allow_duplicates=True),
        ]

    def save(self, file: str):
        """ save training state to file """
        CheckpointCallback.save(file_path=file, pl_module=self.method, update_dict=self.get_checkpoint_update_dict())

    def load(self, file: str) -> bool:
        """ load training state from file """
        checkpoint = CheckpointCallback.load_last_checkpoint(save_dir=file, pl_module=self.method)
        self._load_state_dict(checkpoint.get('trainer_state', {}))
        return len(checkpoint) > 0

    def train_until_max_epoch(self):
        """ train all remaining epochs """
        self.train_until_epoch(self.max_epoch)

    def train_until_epoch(self, epoch: int) -> 'AbstractTrainer':
        try:
            if epoch >= self.stop_epoch > 0:
                epoch = min([epoch, self.stop_epoch])
                self.logger.info('Will stop training early after %d epochs (scheduler, regularizer, etc. '
                                 'are subject to max_epochs=%d)!' % (self.stop_epoch, self.max_epoch))
            rem_epochs = epoch - self.method.trained_epochs
            if rem_epochs > 0:
                self.train_epochs(rem_epochs)
            else:
                self.logger.info('Already trained %d epochs! (task: train to %d)' % (self.method.trained_epochs, epoch))
            return self
        finally:
            self.cleanup()

    def train_epochs(self, epochs=1, run_eval=True, run_test=True):
        """ train 'epochs' epochs """
        raise NotImplementedError

    def eval_epoch(self):
        """ eval one epoch """
        raise NotImplementedError

    def test_epoch(self):
        """ test one epoch """
        raise NotImplementedError

    def get_optimizers(self) -> [Optimizer]:
        """ get optimizers """
        raise NotImplementedError

    def get_schedulers(self) -> [AbstractScheduler]:
        """ get schedulers """
        raise NotImplementedError

    def cleanup(self):
        self.resource_logger.stop()
