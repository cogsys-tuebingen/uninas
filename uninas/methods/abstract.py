from collections import defaultdict
import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from torch.cuda.amp import autocast, GradScaler
from uninas.data.abstract import AbstractBatchAugmentation, AbstractDataSet
from uninas.networks.abstract import AbstractNetwork
from uninas.training.result import EvalLogResult, TrainLogResult
from uninas.training.criteria.common import AbstractCriterion
from uninas.training.regularizers.abstract import AbstractRegularizer
from uninas.training.optimizers.abstract import MultiOptimizer
from uninas.utils.args import Argument, MetaArgument, ArgsInterface, Namespace, find_in_args
from uninas.utils.misc import split
from uninas.utils.loggers.python import get_logger
from uninas.register import Register

logger = get_logger()


class AbstractMethod(pl.LightningModule, ArgsInterface):

    def __init__(self, hparams: Namespace):
        super().__init__()
        ArgsInterface.__init__(self)
        assert isinstance(hparams, Namespace)
        self.hparams = hparams
        for k, v in vars(self.hparams).items():
            assert isinstance(v, (int, float, str, bool)), 'Namespace argument "%s" is of type %s' % (k, type(v))

        # data
        self.data_set = self._parsed_meta_argument('cls_data', self.hparams, index=None)(self.hparams)

        # model
        _, self.max_epochs = find_in_args(self.hparams, '.max_epochs')
        self.strategy = self.get_strategy()
        self.net = self._get_new_network()
        self.net.build(s_in=self.data_set.get_data_shape(), s_out=self.data_set.get_label_shape())

        # criterion
        weights, self.criterion = self.get_weights_criterion()

        # weight initializers
        initializers = self.init_multiple(hparams, 'cls_initializers')
        assert (not self.net.has_loaded_weights()) or (len(initializers) == 0),\
            "Using weight initializers on a network with pre-trained weights!"
        for initializer in initializers:
            initializer.initialize_weights(self.net)

        # metrics, regularizers
        self.metrics = [m(hparams, i, weights)
                        for i, m in enumerate(self._parsed_meta_arguments('cls_metrics', self.hparams, index=None))]
        self.regularizers = self.init_multiple(hparams, 'cls_regularizers')

        # epoch stats
        self._current_epoch = 0
        self.trained_epochs = 0
        self._time_start = time.time()
        self._time_epoch_start = None

        # classes, since Register does not work with ddp anymore
        self._cls_optimizers = self._parsed_meta_arguments('cls_optimizers', self.hparams, index=None)
        self._cls_schedulers = self._parsed_meta_arguments('cls_schedulers', self.hparams, index=None)

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
            MetaArgument('cls_metrics', metrics, help_name='training metric'),
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
        return self._parsed_meta_argument('cls_network', self.hparams, index=None).from_args(self.hparams)

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

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx, key='train', **net_kwargs) -> TrainLogResult:
        loss, dct = self._generic_step(batch, batch_idx, key, self.data_set.train_batch_augmentations, **net_kwargs)
        return TrainLogResult(loss, dct)

    def validation_step(self, batch, batch_idx, key='val', **net_kwargs) -> EvalLogResult:
        loss, dct = self._generic_step(batch, batch_idx, key, self.data_set.train_batch_augmentations, **net_kwargs)
        return EvalLogResult(loss, dct)

    def test_step(self, batch, batch_idx, key='test', **net_kwargs) -> EvalLogResult:
        loss, dct = self._generic_step(batch, batch_idx, key, self.data_set.train_batch_augmentations, **net_kwargs)
        return EvalLogResult(loss, dct)

    @classmethod
    def _generic_data(cls, batch, batch_augments: AbstractBatchAugmentation) -> (torch.Tensor, {str: torch.Tensor}):
        inputs, targets = batch
        if batch_augments is not None:
            with torch.no_grad():
                inputs, targets = batch_augments(inputs, targets)
        return inputs, targets

    def _generic_step(self, batch, batch_idx, key: str, batch_augments: AbstractBatchAugmentation, **net_kwargs)\
            -> (torch.Tensor, {str: torch.Tensor}):
        inputs, targets = self._generic_data(batch, batch_augments)
        with self.amp_autocast:
            logits = self(inputs, **net_kwargs)
        loss = self._loss(logits, targets)
        with torch.no_grad():
            dct = {key + '/loss': loss.clone().detach(), 'num': inputs.size(0)}
            for metric in self.metrics:
                dct.update(metric.evaluate(self.net, inputs, logits, targets, key))
        if self.strategy is not None:
            self.strategy.feedback(key, dct, self.current_epoch, batch_idx)
        return self.amp_scaler.scale(loss), dct

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """ un-squeezed loss value """
        return self.criterion(logits, targets).unsqueeze(0)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
                       clip_value=-1, clip_norm_value=-1, clip_norm_type=-1, **__):
        optimizer.clip_grad(self.amp_scaler, clip_value, clip_norm_value, clip_norm_type)
        self.amp_scaler.step(optimizer)
        self.amp_scaler.update()
        optimizer.zero_grad()

    def optimizer_zero_grad(self, *_, **__):
        pass

    @classmethod
    def _mean_all(cls, outputs: [EvalLogResult]) -> EvalLogResult:
        """ average all key-value pairs in the outputs dicts """
        if isinstance(outputs, EvalLogResult):
            return outputs
        if len(outputs) == 0:
            return EvalLogResult()
        dct = defaultdict(list)
        dct_num = defaultdict(int)
        for o in outputs:
            log_info = o.get_log_info()
            num = log_info['num'].item()
            # loss
            dct['loss'].append(o.minimize if o.minimize is not None else o.checkpoint_on)
            dct_num['loss'] += num
            # other metrics
            for k, v in log_info.items():
                if k == 'num':
                    pass
                else:
                    dct[k].append(v * num)
                    dct_num[k] += num
        loss = torch.cat(dct.pop('loss'), dim=0).sum().detach_().cpu() / dct_num['loss']
        return EvalLogResult(loss, {k: torch.cat(v, dim=0).sum().detach_() / dct_num[k] for k, v in dct.items()})

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        self._current_epoch = self.trained_epochs
        self.trained_epochs += 1

    def validation_epoch_end(self, outputs: list) -> EvalLogResult:
        return self._mean_all(outputs)

    def test_epoch_end(self, outputs: list) -> EvalLogResult:
        return self._mean_all(outputs)

    def on_epoch_start(self, log=True) -> dict:
        log_dict = self._on_epoch_start()
        if log and len(log_dict) > 0:
            self.log_metrics(log_dict)
        return log_dict

    def _on_epoch_start(self) -> dict:
        self._time_epoch_start = time.time()
        log_dict = {}
        # for the first epoch
        if self.current_epoch == 0:
            for reg in self.get_regularizers():
                log_dict.update(reg.on_start(self.max_epochs, self.net))
        # strategy
        if self.strategy is not None:
            self.strategy.on_epoch_start(self.current_epoch)
        # regularizers
        for reg in self.get_regularizers():
            log_dict.update(reg.on_epoch_start(self.current_epoch, self.max_epochs, self.net))
        return log_dict

    def on_epoch_end(self, log=True) -> dict:
        log_dict = self._on_epoch_end()
        if log and len(log_dict) > 0:
            self.log_metrics(log_dict)
        self._current_epoch = self.trained_epochs
        return log_dict

    def _on_epoch_end(self) -> dict:
        log_dict = {}
        # strategy
        if self.strategy is not None:
            self._is_finished = self._is_finished or self.strategy.on_epoch_end(self.current_epoch)
            log_dict = self._add_to_dict(log_dict, self.strategy.highest_value_per_weight(), suffix='max_weight')
        # regularizers
        for reg in self.get_regularizers():
            log_dict.update(reg.on_epoch_end(self.current_epoch, self.max_epochs, self.net))
        # time
        t = time.time()
        log_dict.update({
            'time/total': t - self._time_start,
            'time/epoch': t - self._time_epoch_start,
        })
        return log_dict

    def log_hyperparams(self):
        # the LightningTrainer does this automatically
        self.logger.log_hyperparams(self.hparams)

    def log_metrics(self, log_dict: dict):
        self.logger.log_metrics(log_dict, step=self.current_epoch)

    def flush_logging(self):
        if self.logger is not None:
            self.logger.finalize("success")

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

    # ---------------------------------------------------------------------------------------------------------------- #
    # data, optimizers, schedulers, criterion
    # ---------------------------------------------------------------------------------------------------------------- #

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return self.data_set.train_loader(dist=False)

    def val_dataloader(self):
        return self.data_set.valid_loader(dist=False)

    def test_dataloader(self):
        return self.data_set.test_loader(dist=False)

    def configure_optimizers(self) -> (list, list):
        """ get optimizers/schedulers """
        assert len(self._cls_optimizers) == 1
        optimizer = self._cls_optimizers[0](self.hparams, named_params=self.net.named_parameters(), index=0)
        assert len(self._cls_schedulers) <= 1
        if len(self._cls_schedulers) == 1:
            return [optimizer], [self._cls_schedulers[0](self.hparams, optimizer, self.max_epochs, index=0)]
        return [optimizer], []

    def get_weights_criterion(self) -> (list, AbstractCriterion):
        weights = self.net.get_head_weightings()
        cls_criterion = self._parsed_meta_argument('cls_criterion', self.hparams, index=None)
        criterion = cls_criterion(weights, self.hparams, self.data_set)
        if len(weights) > 1:
            logger.info("Weighting model heads: %s" % str(weights))
        return weights, criterion

    # ---------------------------------------------------------------------------------------------------------------- #
    # network
    # ---------------------------------------------------------------------------------------------------------------- #

    def profile_macs(self) -> np.int64:
        """ profile the macs for a single forward pass on a single data point """
        is_training = self.net.training
        if is_training:
            self.net.eval()
        macs = self.net.profile_macs(self.data_set.sample_random_data(batch_size=1).to(self.net.get_device()))
        if is_training:
            self.net.train()
        return macs

    def get_strategy(self):
        """ get strategy for architecture weights """
        return None

    def save_configs(self, cfg_dir: str):
        os.makedirs(cfg_dir, exist_ok=True)
        if self.strategy is not None:
            Register.builder.save_config(self.net.config(finalize=False), '%s/search.network_config' % cfg_dir)
            Register.builder.save_config(self.net.config(finalize=True), '%s/finalized.network_config' % cfg_dir)
        else:
            Register.builder.save_config(self.net.config(finalize=True), '%s/network.network_config' % cfg_dir)


class AbstractOptimizationMethod(AbstractMethod):
    """
    train network and architecture weights with the same optimizer
    """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        self.update_architecture_weights = True
        self.train_loader = None

        # mask
        for idx in split(self._parsed_argument('mask_indices', self.hparams), cast_fun=int):
            self.strategy.mask_index(idx)
            logger.info("Globally masking arc choices with index %d" % idx)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('mask_indices', default="", type=str, help='[int] mask specific primitives from being used'),
        ]


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

    def optimizer_step(self, epoch, batch_idx, optimizer: MultiOptimizer, optimizer_idx, second_order_closure=None,
                       clip_value=-1, clip_norm_value=-1, clip_norm_type=-1, **__):
        optimizer.clip_grad(self.opt_idx, self.amp_scaler, clip_value, clip_norm_value, clip_norm_type)
        self.amp_scaler.step(optimizer.at_index(self.opt_idx))
        self.amp_scaler.update()
        optimizer.zero_grad_all()

    def training_step(self, batch, batch_idx, key='train', **net_kwargs) -> TrainLogResult:
        self.opt_idx, real_batch = batch
        key = key + ['/net', '/arc'][self.opt_idx]
        loss, dct = self._generic_step(real_batch, batch_idx, key, self.data_set.train_batch_augmentations, **net_kwargs)
        return TrainLogResult(loss, dct)

    def train_dataloader(self):
        self.train_loader = self.data_set.interleaved_train_valid_loader(multiples=(1, 1))
        return self.train_loader

    def val_dataloader(self):
        return None

    def configure_optimizers(self) -> ([MultiOptimizer], list):
        """ get optimizers/schedulers """
        optimizers, schedulers = [], []
        named_params = self.net.named_net_arc_parameters()
        for i in range(2):
            optimizers.append(self._cls_optimizers[i](self.hparams, index=i, named_params=named_params[i]))
            if len(self._cls_schedulers) > i:
                scheduler = self._cls_schedulers[i](self.hparams, optimizers[-1], self.max_epochs, index=i)
                if scheduler is not None:
                    schedulers.append(scheduler)
        assert len(schedulers) <= len(optimizers) == 2
        return [MultiOptimizer(optimizers)], schedulers

    def set_loader_multiples(self, multiples=(1, 1)):
        if self.train_loader is not None:
            self.train_loader.set_multiples(multiples)
