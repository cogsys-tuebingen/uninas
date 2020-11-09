from typing import Union
import torch
from torch.optim.optimizer import Optimizer
from uninas.methods.abstract import AbstractMethod
from uninas.training.trainer.abstract2 import AbstractTrainer
from uninas.training.result import EvalLogResult, TrainLogResult
from uninas.utils.args import Namespace
from uninas.utils.torch.misc import itemize
from uninas.utils.torch.ema import ModelEMA
from uninas.register import Register


@Register.trainer()
class SimpleTrainer(AbstractTrainer):
    """
    A simple trainer for a single device (GPU)
    """

    def __init__(self, method: AbstractMethod, args: Namespace, *_, **__):
        # preparation for 'set_method'
        self.method_ema = None
        self.optimizers, self.schedulers = [], []
        self.loader_train, self.loader_eval, self.loader_test = None, None, None

        super().__init__(method, args, *_, **__)
        assert self.mover.get_num_devices() == 1, 'Can only use this trainer on a single gpu'
        assert len(self.optimizers) == 1

    def set_method(self, method: AbstractMethod):
        """ give the trainer a method to optimize """
        self.method = self.mover.move_module(method)
        assert isinstance(self.method, AbstractMethod)
        self.method_ema = ModelEMA.maybe_init(self.logger, self.method, self.ema_decay, self.ema_device)
        self.optimizers, self.schedulers = self.method.configure_optimizers()
        self.method.logger = self.exp_logger

        self.loader_train = self.method.train_dataloader()
        self.loader_eval = self.method.val_dataloader()
        self.loader_test = self.method.test_dataloader()

    def _get_state_dict(self) -> dict:
        """ get the internal state """
        return {
            'optimizers': [o.state_dict() for o in self.optimizers],
            'schedulers': [s.state_dict() for s in self.schedulers],
        }

    def _load_state_dict(self, state: dict):
        """ load the internal state """
        for o, dct in zip(self.optimizers, state.get('optimizers', [])):
            o.load_state_dict(dct)
        for s, dct in zip(self.schedulers, state.get('schedulers', [])):
            s.load_state_dict(dct)

    def log_dict(self, log_dict: {str: torch.Tensor}):
        for k, v in log_dict.items():
            self.logger.info('    {:<30}{}'.format(k, itemize(v)))
        self.method.log_metrics(log_dict)

    def summarize_results(self, results: [Union[EvalLogResult, TrainLogResult]]) -> EvalLogResult:
        return self.method.validation_epoch_end(results)

    def _next_batch(self, loader) -> list:
        """ get the next batch of the loader, move all tensors to the gpu device """
        return self.mover.move(loader.__next__())

    def train_epochs(self, epochs=1, run_eval=True, run_test=True, **net_kwargs):
        """ train 'epochs' epochs """
        if self.loader_train is not None:
            for c in self.callbacks:
                c.setup(self, self.method, "fit+test")
            self.resource_logger.wakeup()
            num_steps = self.num_test_steps if self.is_test_run else len(self.loader_train)
            is_finished = False

            for i in range(epochs):
                self.logger.info('Training, epoch %d, %d steps' % (self.method.current_epoch, num_steps))
                if is_finished:
                    self.logger.info('The method finished, stopping early.')
                    break

                # log regularizers, the learning rate, ...
                log_dict = self.method.on_epoch_start(log=False)
                log_dict.update(self.get_optimizer_log_dict())
                self.method.log_metrics(log_dict)
                for c in self.callbacks:
                    c.on_train_epoch_start(self, self.method, self.method_ema, log_dict=log_dict)

                # train steps, log info, callbacks
                log_dict = self.train_steps(num_steps, **net_kwargs)
                self.log_dict(log_dict)
                self.method.training_epoch_end([])
                if self.method_ema is not None:
                    self.method_ema.update(self.method)
                for c in self.callbacks:
                    c.on_train_epoch_end(self, self.method, self.method_ema, log_dict=log_dict)

                # step the schedulers
                for scheduler in self.schedulers:
                    scheduler.step()

                # eval / test in the last few epochs of training
                if run_eval and (epochs - i <= self.eval_last or self.eval_last < 0 or is_finished):
                    self.eval_epoch()
                if run_test and (epochs - i <= self.test_last or self.test_last < 0 or is_finished):
                    self.test_epoch()

                self.method.on_epoch_end()
                is_finished = self.method.is_finished()

            for c in self.callbacks:
                c.teardown(self, self.method, "fit+test")

    def train_steps(self, steps=1, **net_kwargs) -> dict:
        """ train 'steps' steps, return the method's log dict """
        if self.loader_train is not None and steps > 0:
            self.method.train()
            results = []
            n = self.num_opt_steps(self.loader_train, 1, self.is_test_run)
            for i in range(steps):
                result = self.method.training_step(batch=self._next_batch(self.loader_train), batch_idx=i, **net_kwargs)
                result.minimize.backward()
                results.append(result)
                self.method.optimizer_step(epoch=self.method.current_epoch, batch_idx=i,
                                           optimizer=self.optimizers[0], optimizer_idx=0,
                                           clip_value=self.cg_v, clip_norm_value=self.cg_nv, clip_norm_type=self.cg_nt)
                for scheduler in self.schedulers:
                    scheduler.step_samples(n=n)
            return self.summarize_results(results).get_log_info()
        return {}

    def eval_epoch(self, **net_kwargs):
        """ eval one epoch """
        if self.loader_eval is not None:
            self.resource_logger.wakeup()
            num_steps = self.num_test_steps if self.is_test_run else len(self.loader_eval)
            self.logger.info('Eval, epoch %d, %d steps' % (self.method.current_epoch, num_steps))
            log_dict = self.eval_steps(num_steps, **net_kwargs)
            self.log_dict(log_dict)
            for c in self.callbacks:
                c.on_validation_epoch_end(self, self.method, self.method_ema, log_dict=log_dict)

    def eval_steps(self, steps=1, **net_kwargs) -> dict:
        """ eval 'steps' steps, return the method's log dict """
        main_dict = {}
        with torch.no_grad():
            if self.loader_eval is not None and steps > 0:
                for m, fmt in self.iterate_usable_methods(self.method, self.method_ema):
                    m.eval()
                    results = []
                    for i in range(steps):
                        results.append(m.validation_step(batch=self._next_batch(self.loader_eval), batch_idx=i, **net_kwargs))
                    # add suffix to the string before the first /
                    for k, v in self.summarize_results(results).get_log_info().items():
                        ks = k.split('/')
                        ks[0] = fmt % ks[0]
                        main_dict['/'.join(ks)] = v
            return main_dict

    def test_epoch(self, **net_kwargs):
        """ test one epoch """
        if self.loader_test is not None:
            self.resource_logger.wakeup()
            num_steps = self.num_test_steps if self.is_test_run else len(self.loader_test)
            self.logger.info('Test, epoch %d, %d steps' % (self.method.current_epoch, num_steps))
            log_dict = self.test_steps(num_steps, **net_kwargs)
            self.log_dict(log_dict)
            for c in self.callbacks:
                c.on_test_epoch_end(self, self.method, self.method_ema, log_dict=log_dict)

    def test_steps(self, steps=1, **net_kwargs) -> dict:
        """ test 'steps' steps, return the method's log dict """
        main_dict = {}
        with torch.no_grad():
            if self.loader_test is not None and steps > 0:
                for m, fmt in self.iterate_usable_methods(self.method, self.method_ema):
                    m.eval()
                    results = []
                    for i in range(steps):
                        results.append(m.test_step(batch=self._next_batch(self.loader_test), batch_idx=i, **net_kwargs))
                    # add suffix to the string before the first /
                    for k, v in self.summarize_results(results).get_log_info().items():
                        ks = k.split('/')
                        ks[0] = fmt % ks[0]
                        main_dict['/'.join(ks)] = v
            return main_dict

    def forward_steps(self, steps=1, **net_kwargs):
        """ have 'steps' forward passes on the training set without gradients, e.g. to sanitize batch-norm stats """
        if self.loader_train is not None and steps > 0:
            self.method.train()
            with torch.no_grad():
                for i in range(steps):
                    self.method.training_step(batch=self._next_batch(self.loader_train), batch_idx=i, **net_kwargs)

    def get_optimizers(self) -> [Optimizer]:
        """ get optimizers """
        return self.optimizers
