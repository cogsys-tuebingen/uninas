from collections import defaultdict
from typing import Union, Iterable, Optional, Callable
import os
import time
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.cuda.amp import autocast, GradScaler
from uninas.data.abstract import AbstractBatchAugmentation, AbstractDataSet
from uninas.models.networks.abstract import AbstractNetwork
from uninas.methods.strategy_manager import StrategyManager
from uninas.training.result import LogResult
from uninas.training.criteria.abstract import MultiCriterion
from uninas.training.initializers.abstract import AbstractInitializer
from uninas.training.metrics.abstract import AbstractMetric
from uninas.training.regularizers.abstract import AbstractRegularizer
from uninas.training.optimizers.abstract import AbstractOptimizerFunctions, WrappedOptimizer, MultiWrappedOptimizer
from uninas.training.schedulers.abstract import AbstractScheduler
from uninas.training.result import ResultValue
from uninas.utils.args import Argument, MetaArgument, ArgsInterface, Namespace, find_in_args
from uninas.utils.loggers.python import LoggerManager, Logger, log_in_columns
from uninas.utils.loggers.exp import LightningLoggerBase
from uninas.utils.torch.decorators import use_eval
from uninas.utils.torch.loader import CustomIterator
from uninas.register import Register


class AbstractMethod(pl.LightningModule, ArgsInterface):
    """
    Many of the non-specialized things that a network training entails
    - network, initialization
    - data and augmentations
    - regularization
    - criterion
    - possibly behaviour for architecture search
    - metrics
    - optimizer, scheduler
    """

    _key_forward = 'forward'
    _key_train = 'train'
    _key_val = 'val'
    _key_test = 'test'

    def __init__(self, hparams: Namespace):
        super().__init__()
        ArgsInterface.__init__(self)
        assert isinstance(hparams, Namespace)
        self.save_hyperparameters(hparams)
        for k, v in vars(self.hparams).items():
            assert isinstance(v, (int, float, str, bool)), 'Namespace argument "%s" is of type %s' % (k, type(v))

        # logging, has to be set by the trainer
        self._logger = None

        # data
        data_set_cls = self._parsed_meta_argument(Register.data_sets, 'cls_data', self.hparams, index=None)
        self.data_set = data_set_cls.from_args(self.hparams, index=None)
        assert isinstance(self.data_set, AbstractDataSet)

        # model
        _, self.max_epochs = find_in_args(self.hparams, '.max_epochs')
        self.strategy_manager = self.setup_strategy()
        self.net = self._get_new_network()
        assert isinstance(self.net, AbstractNetwork)
        self.net.build(s_in=self.data_set.get_data_shape(), s_out=self.data_set.get_label_shape())

        # method forward function
        self._forward_fun = 0
        self._forward_mode_names = {
            'default': 0,
            'custom': 1,
        }

        # criterion
        weights, self.criterion = self.get_weights_criterion()
        assert isinstance(self.criterion, MultiCriterion)

        # weight initializers
        initializers = self.init_multiple(Register.initializers, hparams, 'cls_initializers')
        assert (not self.net.has_loaded_weights()) or (len(initializers) == 0),\
            "Using weight initializers on a network with pre-trained weights!"
        for initializer in initializers:
            assert isinstance(initializer, AbstractInitializer)
            initializer.initialize_weights(self.net)

        # metrics, regularizers
        cls_metrics = self._parsed_meta_arguments(Register.metrics, 'cls_metrics', self.hparams, index=None)
        self.metrics = [m.from_args(hparams, i, self.data_set, weights) for i, m in enumerate(cls_metrics)]
        assert all([isinstance(m, AbstractMetric) for m in self.metrics])
        self.regularizers = self.init_multiple(Register.regularizers, hparams, 'cls_regularizers')
        assert all([isinstance(r, AbstractRegularizer) for r in self.regularizers])

        # optimizer/scheduler classes, since Register does not work with ddp
        self._cls_optimizers = self._parsed_meta_arguments(Register.optimizers, 'cls_optimizers', self.hparams, index=None)
        self._cls_schedulers = self._parsed_meta_arguments(Register.schedulers, 'cls_schedulers', self.hparams, index=None)
        assert all([issubclass(cls, AbstractOptimizerFunctions) for cls in self._cls_optimizers])
        assert all([issubclass(cls, AbstractScheduler) for cls in self._cls_schedulers])

        # epoch stats
        self._current_epoch = 0
        self.trained_epochs = 0
        self._time_start = time.time()
        self._time_epoch_start = None

        # possibly early stopping
        self._is_finished = False

        # automatic mixed precision
        amp_enabled = self._parsed_argument('amp_enabled', hparams)
        self.amp_autocast = autocast(enabled=amp_enabled)
        self.amp_scaler = GradScaler(enabled=amp_enabled)

    @property
    def current_epoch(self) -> int:
        if self.trainer is None:
            return self._current_epoch
        return super().current_epoch

    def get_network(self) -> AbstractNetwork:
        return self.net

    def get_regularizers(self) -> [AbstractRegularizer]:
        """ get regularizers """
        return self.regularizers

    def get_data_set(self) -> AbstractDataSet:
        """ get data set """
        assert isinstance(self.data_set, AbstractDataSet)
        return self.data_set

    @classmethod
    def meta_args_to_add(cls, num_optimizers=1, search=True) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        kwargs = Register.get_my_kwargs(cls)
        metrics = Register.metrics.filter_match_all(distill=kwargs.get('distill'))
        criteria = Register.criteria.filter_match_all(distill=kwargs.get('distill'))
        networks = Register.networks.filter_match_all(search=search)

        return super().meta_args_to_add() + [
            MetaArgument('cls_data', Register.data_sets, help_name='data set', allowed_num=1),
            MetaArgument('cls_network', networks, help_name='network', allowed_num=1),
            MetaArgument('cls_criterion', criteria, help_name='criterion', allowed_num=1),
            MetaArgument('cls_metrics', metrics, help_name='training metric', allow_duplicates=True),
            MetaArgument('cls_initializers', Register.initializers, help_name='weight initializer'),
            MetaArgument('cls_regularizers', Register.regularizers, help_name='regularizer'),
            MetaArgument('cls_optimizers', Register.optimizers, help_name='optimizer', allow_duplicates=True, allowed_num=num_optimizers, use_index=True),
            MetaArgument('cls_schedulers', Register.schedulers, help_name='scheduler', allow_duplicates=True, allowed_num=(0, num_optimizers), use_index=True),
        ]

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('amp_enabled', default="False", type=str, help='use AMP (automatic mixed precision)', is_bool=True),
        ]

    def _get_new_network(self) -> AbstractNetwork:
        return self._parsed_meta_argument(Register.networks, 'cls_network', self.hparams, index=None).from_args(self.hparams)

    def on_save_checkpoint(self, checkpoint: dict):
        checkpoint['trained_epochs'] = self.trained_epochs
        checkpoint['net_add_state'] = self.net.save_to_state_dict()

    def on_load_checkpoint(self, checkpoint: dict):
        self._current_epoch = self.trained_epochs = checkpoint.get('trained_epochs', 0)
        self.net.load_from_state_dict(checkpoint.get('net_add_state', dict()))
        self.net.loaded_weights()

    # ---------------------------------------------------------------------------------------------------------------- #
    # training and logging
    # ---------------------------------------------------------------------------------------------------------------- #

    def set_logger(self, logger: LightningLoggerBase):
        self._logger = logger

    @property
    def logger(self) -> Union[LightningLoggerBase, None]:
        """ Reference to the logger object in the Trainer. """
        return self.trainer.logger if self.trainer else self._logger

    @property
    def exp_logger(self) -> Union[LightningLoggerBase, None]:
        """ Reference to the logger object in the Trainer. """
        return self.logger

    def use_forward_mode(self, mode='default'):
        """
        :param mode:
            default: default pass from input to all head outputs
            custom: some methods may require custom training (e.g. (self-)distillation)
        """
        v = self._forward_mode_names.get(mode)
        assert v is not None, "unknown mode %s" % mode
        self._forward_fun = v

    def forward(self, *args, **kwargs):
        if self._forward_fun == 0:
            return self.forward_default(*args, **kwargs)
        return self.forward_custom(*args, **kwargs)

    def forward_default(self, x: torch.Tensor) -> [torch.Tensor]:
        return self.net(x)

    def forward_custom(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def forward_step(self, batch: (torch.Tensor, torch.Tensor), batch_idx: int, **net_kwargs) -> LogResult:
        """
        forward step, to correct batchnorm statistics for a sub-network

        :param batch: (inputs, outputs)
        :param batch_idx:
        :param net_kwargs:
        :return:
        """
        assert self.training, "The network must be in training mode here"
        with torch.no_grad():
            loss, dct = self._generic_step(
                batch=batch, batch_idx=batch_idx, key=self._key_forward,
                batch_augments=self.data_set.valid_batch_augmentations, only_losses=False, **net_kwargs)
            return LogResult(loss, dct).detach()

    def training_step(self, batch: (torch.Tensor, torch.Tensor), batch_idx: int, only_losses=False, **net_kwargs)\
            -> LogResult:
        """
        training step, compute result that enables backpropagation

        :param batch: (inputs, outputs)
        :param batch_idx:
        :param only_losses: do not compute metrics or give the architecture strategies feedback
        :param net_kwargs:
        :return:
        """
        assert self.training, "The network must be in training mode here"
        loss, dct = self._generic_step(
            batch=batch, batch_idx=batch_idx, key=self._key_train,
            batch_augments=self.data_set.train_batch_augmentations, only_losses=False, **net_kwargs)
        return LogResult(loss, dct)

    def validation_step(self, batch: (torch.Tensor, torch.Tensor), batch_idx: int, **net_kwargs) -> LogResult:
        """
        validation step

        :param batch: (inputs, outputs)
        :param batch_idx:
        :param net_kwargs:
        :return:
        """
        assert not self.training, "The network must not be in training mode here"
        with torch.no_grad():
            loss, dct = self._generic_step(
                batch=batch, batch_idx=batch_idx, key=self._key_val,
                batch_augments=self.data_set.valid_batch_augmentations, only_losses=False, **net_kwargs)
            return LogResult(loss, dct).detach()

    def test_step(self, batch: (torch.Tensor, torch.Tensor), batch_idx: int, **net_kwargs) -> LogResult:
        """
        test step

        :param batch: (inputs, outputs)
        :param batch_idx:
        :param net_kwargs:
        :return:
        """
        assert not self.training, "The network must not be in training mode here"
        with torch.no_grad():
            loss, dct = self._generic_step(
                batch=batch, batch_idx=batch_idx, key=self._key_test,
                batch_augments=self.data_set.test_batch_augmentations, only_losses=False, **net_kwargs)
            return LogResult(loss, dct).detach()

    @classmethod
    def _generic_data(cls, batch: (torch.Tensor, torch.Tensor), batch_augments: Optional[AbstractBatchAugmentation])\
            -> (torch.Tensor, torch.Tensor):
        """
        unwrap the batch, possibly apply batch augmentations

        :param batch: (inputs, outputs)
        :param batch_augments:
        :return: inputs, targets
        """
        assert isinstance(batch, (tuple, list)), "The batch must be a tuple/list of length 2 (inputs, outputs)"
        assert len(batch) == 2, "The batch must be a tuple of length 2 (inputs, outputs)"
        inputs, targets = batch
        if batch_augments is not None:
            with torch.no_grad():
                inputs, targets = batch_augments(inputs, targets)
        return inputs, targets

    def _generic_step(self, batch: (torch.Tensor, torch.Tensor), batch_idx: int, key: str,
                      batch_augments: Optional[AbstractBatchAugmentation], only_losses=False, **net_kwargs)\
            -> (torch.Tensor, {str: torch.Tensor}):
        """
        generic forward pass

        :param batch: (inputs, outputs)
        :param batch_idx:
        :param key: train/val/test
        :param batch_augments:
        :param only_losses: do not compute metrics or give the architecture strategies feedback
        :param net_kwargs:
        :return: network output, log dict
        """
        # compute outputs
        log_dct = {}
        inputs, targets = self._generic_data(batch, batch_augments)
        with self.amp_autocast:
            logits = self(inputs, **net_kwargs)

        # compute loss
        loss = self._loss(logits, targets)
        if self.strategy_manager is not None:
            losses = self.strategy_manager.get_losses(clear=True)
            if len(losses) > 0:
                log_dct['%s/loss/criterion' % key] = ResultValue(loss.clone().detach(), inputs.size(0))
            for k, loss_ in losses.items():
                log_dct['%s/loss/%s' % (key, k)] = ResultValue(loss_.clone().detach(), inputs.size(0))
                loss = loss + loss_
        log_dct['%s/loss' % key] = ResultValue(loss.clone().detach(), inputs.size(0))

        if not only_losses:
            # compute metrics, log dict
            with torch.no_grad():
                for metric in self.metrics:
                    values = metric.evaluate(self.net, inputs, logits, targets, key)
                    log_dct.update(values)

            # feedback to the architecture strategies
            if self.strategy_manager is not None:
                self.strategy_manager.feedback(key, log_dct, self.current_epoch, batch_idx)

        return self.amp_scaler.scale(loss), log_dct

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """ un-squeezed loss value """
        return self.criterion(logits, targets).unsqueeze(0)

    def optimizer_step(self, *args, epoch: int = None, optimizer: WrappedOptimizer = None,
                       optimizer_closure: Optional[Callable] = None, **kwargs):
        optimizer.step(closure=optimizer_closure)
        self.amp_scaler.update()
        optimizer.zero_grad()

    def optimizer_zero_grad(self, *_, **__):
        pass

    @classmethod
    def _mean_all(cls, outputs: [LogResult]) -> LogResult:
        """ average all key-value pairs in the outputs dicts """
        if isinstance(outputs, LogResult):
            return outputs
        if len(outputs) == 0:
            return LogResult(None)
        dct_val = defaultdict(list)
        dct_num = defaultdict(int)
        for o in outputs:
            log_info = o.get_log_info()
            for k, v in log_info.items():
                dct_val[k].append(v.get_scaled_value())
                dct_num[k] += v.count
        mean = {k: ResultValue(torch.cat(v, dim=0).sum().detach_() / dct_num[k], dct_num[k]) for k, v in dct_val.items()}
        return LogResult(None, log_info=mean).detach()

    def summarize_outputs(self, outputs: list) -> LogResult:
        return self._mean_all(outputs)

    def training_epoch_end(self, outputs: list) -> None:
        self._current_epoch = self.trained_epochs
        self.trained_epochs += 1

    def on_epoch_start(self, log=True, is_last=False) -> dict:
        """
        when the trainer starts a new epoch
        if the method stops early, the is_last flag will never be True
        """
        self._time_epoch_start = time.time()
        log_dict = {}
        # for the first epoch
        if self.current_epoch == 0:
            for reg in self.get_regularizers():
                log_dict.update(reg.on_start(self.max_epochs, self.net))
        # strategy
        if self.strategy_manager is not None:
            self.strategy_manager.on_epoch_start(self.current_epoch)
        # regularizers
        for reg in self.get_regularizers():
            log_dict.update(reg.on_epoch_start(self.current_epoch, self.max_epochs, self.net))
        # metrics
        for m in self.metrics:
            m.on_epoch_start(self._current_epoch, is_last=is_last)

        # method specific
        self._add_to_dict(log_dict, self._on_epoch_start())

        # finally log
        if log and len(log_dict) > 0:
            self.log_metrics(log_dict)
        return log_dict

    def _on_epoch_start(self) -> dict:
        """ additional method-specific """
        return {}

    def on_epoch_end(self, log=True) -> dict:
        log_dict = {}
        # strategy
        if self.strategy_manager is not None:
            self._is_finished = self._is_finished or self.strategy_manager.on_epoch_end(self.current_epoch)
            log_dict = self._add_to_dict(log_dict, self.strategy_manager.get_log_dict())
        # regularizers
        for reg in self.get_regularizers():
            log_dict.update(reg.on_epoch_end(self.current_epoch, self.max_epochs, self.net))

        # method specific
        self._add_to_dict(log_dict, self._on_epoch_end())

        # time
        t = time.time()
        log_dict.update({
            'time/total': t - self._time_start,
            'time/epoch': t - self._time_epoch_start,
        })
        # finally log, update current epoch
        if log and len(log_dict) > 0:
            self.log_metrics(log_dict)
        self._current_epoch = self.trained_epochs
        return log_dict

    def _on_epoch_end(self) -> dict:
        """ additional method-specific """
        return {}

    def log_hyperparams(self):
        # the LightningTrainer does this automatically
        self.exp_logger.log_hyperparams(self.hparams)
        self.exp_logger.save()

    def log_metrics(self, log_dict: {str: float}):
        """ log metrics to all exp loggers """
        log_dict = {k: v.value if isinstance(v, ResultValue) else v for k, v in log_dict.items()}
        self.exp_logger.log_metrics(log_dict, step=self.current_epoch)
        self.exp_logger.save()

    def log_metric_lists(self, log_dict: {str: Union[Iterable[float], float]}):
        """ log metric lists to all exp loggers, enumerating steps starting from 0 """
        for k, v in log_dict.items():
            if isinstance(v, Iterable):
                for i, vx in enumerate(v):
                    self.exp_logger.agg_and_log_metrics({k: vx}, step=i)
            else:
                self.exp_logger.agg_and_log_metrics({k: v}, step=0)

    def get_accumulated_metric_stats(self, prefix="") -> dict:
        """ get all stats of all metrics that are to be visualized """
        # need to flatten the dict, for e.g. ddp synchronization
        stats = {}
        for m in self.metrics:
            for key in [self._key_train, self._key_val, self._key_test]:
                for k, v in m.get_accumulated_stats(key).items():
                    stats[(m.get_log_name(), key, prefix, k)] = v
        return stats

    def eval_accumulated_metric_stats(self, save_dir: str, stats: dict = None, prefix="") -> dict:
        """ visualize the metrics, de-flatten the dict """
        if stats is None:
            stats = self.get_accumulated_metric_stats(prefix=prefix)
        # need to de-flatten the dict
        all_stats = defaultdict(dict)
        for (name, key, prefix, k), v in stats.items():
            all_stats[(name, key, prefix)][k] = v
        # find related metric for each dict, plot it
        log_dict = {}
        for (name, key, prefix), stats in all_stats.items():
            for m in self.metrics:
                if m.get_log_name() == name:
                    x = m.eval_accumulated_stats("%s/%s" % (save_dir, name),
                                                 key=key, prefix=prefix, epoch=self.current_epoch, stats=stats)
                    log_dict.update(x)
                    break
        return log_dict

    def flush_logging(self):
        if self.exp_logger is not None:
            self.exp_logger.finalize("success")

    def is_finished(self) -> bool:
        return self._is_finished

    def _add_to_dict(self, log_dict: dict, dct: dict, suffix='') -> dict:
        """ update log_dict with dict, additionally add class name as key, and optional key suffix """
        key = self.__class__.__name__
        if len(suffix) > 0:
            key = '%s_%s' % (key, suffix)
        for k, v in dct.items():
            log_dict['%s/%s' % (key, k)] = v
        return log_dict

    def log_detailed(self, logger: Logger):
        rows = [
            ("Method", self.str()),
            ("Data set", self.data_set.str()),
            (" > train", self.data_set.list_train_transforms()),
            (" > valid", self.data_set.list_valid_transforms()),
            (" > test", self.data_set.list_test_transforms()),
            ("Criterion", self.criterion),
        ]

        if len(self.metrics) > 0:
            rows.append(("Metrics", ""))
            for i, x in enumerate(self.metrics):
                rows.append((" (%d)" % i, x.str()))

        if len(self.regularizers) > 0:
            rows.append(("Regularizers", ""))
            for i, x in enumerate(self.regularizers):
                rows.append((" (%d)" % i, x.str()))

        optimizers, schedulers = self.configure_optimizers()
        if len(optimizers) > 0:
            rows.append(("Optimizers", ""))
            for i, x in enumerate(optimizers):
                rows.append((" (%d)" % i, str(x)))
        if len(schedulers) > 0:
            rows.append(("Schedulers", ""))
            for i, x in enumerate(schedulers):
                rows.append((" (%d)" % i, x.str()))
        del optimizers, schedulers

        log_in_columns(logger, rows)

    # ---------------------------------------------------------------------------------------------------------------- #
    # data, optimizers, schedulers, criterion
    # ---------------------------------------------------------------------------------------------------------------- #

    def prepare_data(self):
        pass

    def train_dataloader(self, dist=False) -> Union[CustomIterator, None]:
        """
        get a data loader for the training set
        :param dist: distributed sampling, used by DDP
        :return:
        """
        return self.data_set.train_loader(dist=dist)

    def val_dataloader(self, dist=False) -> Union[CustomIterator, None]:
        """
        get a data loader for the validation set
        :param dist: distributed sampling, used by DDP
        :return:
        """
        return self.data_set.valid_loader(dist=dist)

    def test_dataloader(self, dist=False) -> Union[CustomIterator, None]:
        """
        get a data loader for the test set
        :param dist: distributed sampling, used by DDP
        :return:
        """
        return self.data_set.test_loader(dist=dist)

    def configure_optimizers(self) -> (list, list):
        """ get optimizers/schedulers """
        assert len(self._cls_optimizers) == 1
        optimizer = self._cls_optimizers[0].from_args(self.hparams, 0, self.amp_scaler,
                                                      named_params=self.net.named_parameters())
        assert len(self._cls_schedulers) <= 1
        if len(self._cls_schedulers) == 1:
            return [optimizer], [self._cls_schedulers[0].from_args(self.hparams, optimizer, self.max_epochs, index=0)]
        return [optimizer], []

    def get_weights_criterion(self) -> (list, MultiCriterion):
        weights = self.net.get_head_weightings()
        cls_criterion = self._parsed_meta_argument(Register.criteria, 'cls_criterion', self.hparams, index=None)
        criterion = MultiCriterion.from_args(weights, cls_criterion, self.data_set, self.hparams)
        if len(weights) > 1:
            LoggerManager().get_logger().info("Weighting model heads: %s" % str(weights))
        return weights, criterion

    # ---------------------------------------------------------------------------------------------------------------- #
    # network
    # ---------------------------------------------------------------------------------------------------------------- #

    @use_eval
    def profile_macs(self) -> np.int64:
        """ profile the macs for a single forward pass on a single data point """
        macs = -1
        try:
            macs = self.net.profile_macs()
        except Exception as e:
            LoggerManager().get_logger().error("Failed profiling macs:\n%s\n..." % str(e)[:500])
        return macs

    def setup_strategy(self) -> Union[None, StrategyManager]:
        """ set up the strategy for architecture weights """
        return None

    def save_configs(self, cfg_dir: str):
        os.makedirs(cfg_dir, exist_ok=True)
        if self.strategy_manager is not None:
            Register.builder.save_config(self.net.config(finalize=False), cfg_dir, 'search')
            Register.builder.save_config(self.net.config(finalize=True), cfg_dir, 'finalized')
        else:
            Register.builder.save_config(self.net.config(finalize=True), cfg_dir, 'network')

    # ---------------------------------------------------------------------------------------------------------------- #
    # utils
    # ---------------------------------------------------------------------------------------------------------------- #

    def uses_all_paths(self) -> bool:
        """ whether a forward pass will use all parameters (used for distributed training) """
        return self.strategy_manager.uses_all_paths() if isinstance(self.strategy_manager, StrategyManager) else True


class AbstractOptimizationMethod(AbstractMethod):
    """
    train network and architecture weights with the same optimizer
    """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        self.train_loader = None

        # mask
        for idx in self._parsed_argument('mask_indices', self.hparams, split_=int):
            self.strategy_manager.mask_index(idx)
            LoggerManager().get_logger().info("Globally masking arc choices with index %d" % idx)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('mask_indices', default="", type=str, help='[int] mask specific primitives from being used'),
        ]

    def configure_optimizers(self) -> (list, list):
        """ get optimizers/schedulers """
        assert len(self._cls_schedulers) <= len(self._cls_optimizers) == 1
        net_params, arc_params = self.net.named_net_arc_parameters()
        optimizer = self._cls_optimizers[0].from_args(self.hparams, 0, self.amp_scaler,
                                                      named_params=net_params+arc_params)
        if len(self._cls_schedulers) == 1:
            return [optimizer], [self._cls_schedulers[0].from_args(self.hparams, optimizer, self.max_epochs, index=0)]
        return [optimizer], []


class MethodWrapper:
    """
    A wrapper for a Method that enables simply using forward(...) while referring to the correct step(...) functions
    """

    def __init__(self, method: Union[AbstractMethod, None]):
        super().__init__()
        self.method = method
        self._is_train = True
        self._is_valid = False
        self._is_test = False

    def get_method(self) -> AbstractMethod:
        assert isinstance(self.method, AbstractMethod)
        return self.method

    def set_mode(self, train=False, valid=False, test=False):
        assert sum([train, valid, test]) == 1
        self._is_train = train
        self._is_valid = valid
        self._is_test = test
        if train:
            self.method.train()
        else:
            self.method.eval()

    def forward(self, batch, batch_idx: int, **kwargs):
        if self._is_train:
            return self.method.training_step(batch, batch_idx, **kwargs)
        elif self._is_valid:
            return self.method.validation_step(batch, batch_idx, **kwargs)
        elif self._is_test:
            return self.method.test_step(batch, batch_idx, **kwargs)
        raise NotImplementedError


class MethodWrapperModule(MethodWrapper, nn.Module):
    """
    A wrapper for a Method that enables simply using forward(...) while referring to the correct step(...) functions
    """


class AbstractBiOptimizationMethod(AbstractOptimizationMethod):
    """
    train network and architecture weights separately
    """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        self.opt_idx = -1

    @classmethod
    def meta_args_to_add(cls, **__) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add(num_optimizers=2)

    def optimizer_step(self, *args, epoch: int = None, optimizer: MultiWrappedOptimizer = None, **kwargs):
        optimizer.step(index=self.opt_idx)
        self.amp_scaler.update()
        optimizer.zero_grad_all()

    def training_step(self, batch: (int, (torch.Tensor, torch.Tensor)), batch_idx: int, **net_kwargs) -> LogResult:
        """
        training step, compute result that enables backpropagation

        :param batch: (index which optimizer, (inputs, outputs))
        :param batch_idx:
        :param net_kwargs:
        :return:
        """
        assert self.training, "The network must be in training mode here"
        self.opt_idx, real_batch = batch
        key = self._key_train + ['/net', '/arc'][self.opt_idx]
        loss, dct = self._generic_step(real_batch, batch_idx, key, self.data_set.train_batch_augmentations, **net_kwargs)
        return LogResult(loss, dct)

    def train_dataloader(self, dist=False) -> Union[CustomIterator, None]:
        """
        get a data loader for the training and validation set
        :param dist: distributed sampling, used by DDP
        :return:
        """
        self.train_loader = self.data_set.interleaved_train_valid_loader(multiples=(1, 1), dist=dist)
        return self.train_loader

    def val_dataloader(self, dist=False) -> Union[CustomIterator, None]:
        """
        returns None as the validation set is used as part of the training set
        :param dist: distributed sampling, used by DDP
        :return:
        """
        return None

    def configure_optimizers(self) -> ([MultiWrappedOptimizer], list):
        """ get optimizers/schedulers """
        optimizers, schedulers = [], []
        named_params = self.net.named_net_arc_parameters()
        for i in range(2):
            optimizers.append(self._cls_optimizers[i].from_args(self.hparams, i, self.amp_scaler,
                                                                named_params=named_params[i]))
            if len(self._cls_schedulers) > i:
                scheduler = self._cls_schedulers[i].from_args(self.hparams, optimizers[-1], self.max_epochs, index=i)
                if scheduler is not None:
                    schedulers.append(scheduler)
        assert len(schedulers) <= len(optimizers) == 2
        return [MultiWrappedOptimizer(optimizers)], schedulers

    def set_loader_multiples(self, multiples=(1, 1)):
        if self.train_loader is not None:
            self.train_loader.set_multiples(multiples)
