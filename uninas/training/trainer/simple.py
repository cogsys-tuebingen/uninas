import torch
from uninas.methods.abstract_method import AbstractMethod
from uninas.training.trainer.abstract2 import AbstractTrainer
from uninas.training.optimizers.abstract import AbstractOptimizerClosure, Optimizer
from uninas.training.schedulers.abstract import AbstractScheduler
from uninas.training.result import LogResult
from uninas.utils.args import Namespace
from uninas.utils.loggers.python import log_in_columns
from uninas.utils.torch.misc import itemize
from uninas.utils.torch.loader import CustomIterator
from uninas.register import Register


@Register.trainer()
class SimpleTrainer(AbstractTrainer):
    """
    A simple trainer for a single device (e.g. GPU)
    """

    def __init__(self, method: AbstractMethod, args: Namespace, *_, **__):
        # preparation for 'set_method'
        super().__init__(method, args, *_, **__)

        # method and logger
        self.method = self.mover.move_module(self.method)
        assert isinstance(self.method, AbstractMethod)
        self.optimizers, self.schedulers = self.method.configure_optimizers()
        self.optimizer_closures = [o.get_closure(self.mover, self.method.training_step) for o in self.optimizers]
        self.method.set_logger(self.exp_logger)

        # data loaders
        self.loader_train = self.method.train_dataloader(dist=False)
        self.loader_eval = self.method.val_dataloader(dist=False)
        self.loader_test = self.method.test_dataloader(dist=False)

        # initialize clones
        for clone in self.get_method_clones():
            clone.init(self.method)

        assert self.mover.get_num_devices() == 1, 'Can only use this trainer on a single gpu'
        assert len(self.optimizers) == len(self.optimizer_closures) == 1
        assert (not any([isinstance(c, AbstractOptimizerClosure) for c in self.optimizer_closures])) or\
               (self.accumulate_batches == 1), "Can not accumulate batches when optimizer(s) use closures"

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
        rows = [(k, itemize(v)) for k, v in log_dict.items()]
        log_in_columns(self.logger, rows, min_widths=(60, 0), start_space=4)
        self.method.log_metrics(log_dict)

    def summarize_results(self, results: [LogResult]) -> LogResult:
        return self.method.summarize_outputs(results)

    def _next_batch(self, loader: CustomIterator, move=True) -> list:
        """ get the next batch of the loader, move all tensors to the correct device """
        if move:
            return self.mover.move(loader.__next__())
        return loader.__next__()

    def train_epochs(self, epochs=1, run_eval=True, run_test=True, **net_kwargs):
        """ train 'epochs' epochs """
        if self.loader_train is not None:
            self._trigger_callbacks("setup", self, self.method, "fit+test")
            self._trigger_callbacks("on_fit_start", self, self.method)

            self.resource_logger.wakeup()
            num_steps = min([self.num_test_steps, len(self.loader_train)])\
                if self._is_test_run else len(self.loader_train)
            is_finished = False

            for i in range(epochs):
                self.logger.info('Training, epoch %d, %d steps' % (self.method.current_epoch, num_steps))
                if is_finished:
                    self.logger.info('The method finished, stopping early.')
                    break
                self._trigger_callbacks("on_epoch_start", self, self.method)

                # log regularizers, the learning rate, ...
                log_dict = self.method.on_epoch_start(log=False, is_last=(i == epochs-1))
                for clone in self.get_method_clones():
                    clone.get_method().on_epoch_start(log=False, is_last=(i == epochs-1))
                log_dict.update(self.get_optimizer_log_dict())
                self.method.log_metrics(log_dict)
                self._trigger_callbacks("on_train_epoch_start", self, self.method, log_dict=log_dict)

                # train steps, log info, callbacks
                log_dict = self.train_steps(num_steps, **net_kwargs)
                self.method.training_epoch_end([])
                self.log_dict(log_dict)
                for clone in self.get_method_clones():
                    clone.on_training_epoch_end(self.method)
                self._trigger_callbacks("on_train_epoch_end", self, self.method, log_dict=log_dict)

                # step the schedulers
                for scheduler in self.schedulers:
                    scheduler.step()

                # eval / test in the last few epochs of training
                if run_eval and (epochs - i <= self.eval_last or self.eval_last < 0 or is_finished):
                    self.eval_epoch()
                if run_test and (epochs - i <= self.test_last or self.test_last < 0 or is_finished):
                    self.test_epoch()

                # plot/log accumulated metrics
                log_dict = {}
                for method, fmt in self.iterate_methods_on_device():
                    v = method.eval_accumulated_metric_stats(prefix=fmt % 'net', save_dir=self.get_metrics_save_dir("net"))
                    log_dict.update(v)
                if len(log_dict) > 0:
                    self.logger.info('Accumulated stats, epoch %d' % self.method.current_epoch)
                    self.log_dict(log_dict)

                # end epoch
                self.method.on_epoch_end()
                is_finished = self.method.is_finished()
                self._trigger_callbacks("on_epoch_end", self, self.method)

            self._trigger_callbacks("on_fit_end", self, self.method)
            self._trigger_callbacks("teardown", self, self.method, "fit+test")
            for clone in self.get_method_clones():
                clone.stop()

    def train_steps(self, steps=1, **net_kwargs) -> dict:
        """ train 'steps' steps, return the method's log dict """
        if self.loader_train is not None and steps > 0:
            self.method.train()
            results = []
            n = self.num_opt_steps(self.loader_train, 1, self._is_test_run)
            for i in range(steps):
                opt, closure = self.optimizers[0], self.optimizer_closures[0]

                # maybe use closure
                if isinstance(closure, AbstractOptimizerClosure):
                    c = closure.prepare(batch=self._next_batch(self.loader_train, move=False), batch_idx=i, **net_kwargs)
                    self.method.optimizer_step(epoch=self.method.current_epoch, batch_idx=i,
                                               optimizer=opt, optimizer_idx=0, optimizer_closure=c)
                    results.append(c.get_result())
                    for clone in self.get_method_clones():
                        clone.on_update(self.method)

                # default step
                else:
                    result = self.method.training_step(batch=self._next_batch(self.loader_train), batch_idx=i, **net_kwargs)
                    result.backward()
                    result.detach()
                    results.append(result)

                    self._acc_step += 1
                    self._acc_step %= self.accumulate_batches
                    if self._acc_step == 0:
                        self.method.optimizer_step(epoch=self.method.current_epoch, batch_idx=i,
                                                   optimizer=opt, optimizer_idx=0, optimizer_closure=closure)
                        for clone in self.get_method_clones():
                            clone.on_update(self.method)

                for scheduler in self.schedulers:
                    scheduler.step_samples(n=n)
            return self.summarize_results(results).get_log_info()
        return {}

    def eval_epoch(self, **net_kwargs):
        """ eval one epoch """
        if self.loader_eval is not None:
            self._trigger_callbacks("on_validation_epoch_start", self, self.method)
            self.resource_logger.wakeup()
            num_steps = min([self.num_test_steps, len(self.loader_eval)]) if self._is_test_run else len(self.loader_eval)
            self.logger.info('Eval, epoch %d, %d steps' % (self.method.current_epoch, num_steps))
            log_dict = self.eval_steps(num_steps, **net_kwargs)
            self.log_dict(log_dict)
            self._trigger_callbacks("on_validation_epoch_end", self, self.method, log_dict=log_dict)
            return log_dict
        return {}

    def test_epoch(self, **net_kwargs):
        """ test one epoch """
        if self.loader_test is not None:
            self._trigger_callbacks("on_test_epoch_start", self, self.method)
            self.resource_logger.wakeup()
            num_steps = min([self.num_test_steps, len(self.loader_test)]) if self._is_test_run else len(self.loader_test)
            self.logger.info('Test, epoch %d, %d steps' % (self.method.current_epoch, num_steps))
            log_dict = self.test_steps(num_steps, **net_kwargs)
            self.log_dict(log_dict)
            self._trigger_callbacks("on_test_epoch_end", self, self.method, log_dict=log_dict)
            return log_dict
        return {}

    def _eval_or_test_steps(self, steps=1, is_eval=True, **net_kwargs) -> dict:
        """ eval or test 'steps' steps, return the method's log dict """
        main_dict = {}
        loader = self.loader_eval if is_eval else self.loader_test
        with torch.no_grad():
            if loader is not None and steps > 0:
                # get usable methods, set them to eval mode
                methods_fmts = list(self.iterate_methods_on_device())
                funs, results = [], []
                for (m, _) in methods_fmts:
                    m.eval()
                    results.append([])
                    funs.append(m.validation_step if is_eval else m.test_step)
                # iterate num batches, same batch for all methods
                for i in range(steps):
                    batch = self._next_batch(loader)
                    for j in range(len(methods_fmts)):
                        results[j].append(funs[j](batch=batch, batch_idx=i, **net_kwargs))
                # add suffix to the string before the first /
                for r, (_, fmt) in zip(results, methods_fmts):
                    for k, v in self.summarize_results(r).get_log_info().items():
                        ks = k.split('/')
                        ks[0] = fmt % ks[0]
                        main_dict['/'.join(ks)] = v
            return main_dict

    def eval_steps(self, steps=1, **net_kwargs) -> dict:
        """ eval 'steps' steps, return the method's log dict """
        return self._eval_or_test_steps(steps=steps, is_eval=True, **net_kwargs)

    def test_steps(self, steps=1, **net_kwargs) -> dict:
        """ test 'steps' steps, return the method's log dict """
        return self._eval_or_test_steps(steps=steps, is_eval=False, **net_kwargs)

    def forward_steps(self, steps=1, **net_kwargs):
        """ have 'steps' forward passes on the training set without gradients, e.g. to sanitize batch-norm stats """
        if self.loader_train is not None and steps > 0:
            self.method.train()
            with torch.no_grad():
                for i in range(steps):
                    self.method.forward_step(batch=self._next_batch(self.loader_train), batch_idx=i, **net_kwargs)

    def get_optimizers(self) -> [Optimizer]:
        """ get optimizers """
        return self.optimizers

    def get_schedulers(self) -> [AbstractScheduler]:
        """ get schedulers """
        return self.schedulers
