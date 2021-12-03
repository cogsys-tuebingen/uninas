import os
import torch
from datetime import timedelta
from typing import Iterable, Tuple
from torch.nn.parallel import DistributedDataParallel as Ddp
import torch.multiprocessing as mp
import torch.distributed as dist
from uninas.methods.abstract_method import AbstractMethod, MethodWrapperModule
from uninas.training.trainer.abstract import AbstractTrainerFunctions
from uninas.training.trainer.abstract2 import AbstractTrainer
from uninas.training.optimizers.abstract import AbstractOptimizerClosure, Optimizer
from uninas.training.schedulers.abstract import AbstractScheduler
from uninas.training.result import LogResult
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.training.clones.abstract import AbstractMethodClone
from uninas.training.callbacks.abstract import AbstractCallback
from uninas.training.callbacks.checkpoint import CheckpointCallback
from uninas.utils.loggers.exp import LightningLoggerBase
from uninas.utils.loggers.python import log_in_columns
from uninas.utils.torch.misc import itemize
from uninas.utils.torch.loader import CustomIterator
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


class SimpleDDPTrainerTrainEpochsImpl(AbstractTrainerFunctions):
    """
    Train until epoch, ddp
    """

    def __init__(self, rank: int, world_size: int, method: AbstractMethod, method_clones: [AbstractMethodClone],
                 save_dir: str, mover: AbstractDeviceMover, callbacks: [AbstractCallback],
                 exp_logger: LightningLoggerBase, epochs=1, eval_last=2, test_last=2,
                 is_test_run=False, accumulate_batches=1, use_sync_bn=False, load_state: dict = None):
        super().__init__()
        self.rank = rank
        self.save_dir = save_dir
        self.mover = mover.get_device_subset([rank])
        self.mover.set_rank()
        self.logger = SimpleDDPTrainer.get_logger(None, is_test_run, self.save_dir, suffix=str(rank))
        self.accumulate_batches = accumulate_batches
        self._acc_step = 0
        self.is_test_run = is_test_run

        self.ddp_method = None
        self.method_clones = method_clones

        self._setup_ddp(world_size, method, exp_logger, use_sync_bn)

        self.optimizers, self.schedulers = self.get_method().configure_optimizers()
        self.optimizer_closures = [o.get_closure(self.mover, self.get_method_ddp()) for o in self.optimizers]
        self.callbacks = callbacks
        self._trigger_callbacks("setup", self, self.get_method(), "fit+test")
        self._trigger_callbacks("on_fit_start", self, self.get_method())

        if epochs > 0:
            train_loader = self.get_method().train_dataloader(dist=True)
            eval_loader = self.get_method().val_dataloader(dist=True) if eval_last != 0 else None
            test_loader = self.get_method().test_dataloader(dist=True) if test_last != 0 else None

            num_steps = min([SimpleDDPTrainer.num_test_steps, len(train_loader)]) if is_test_run else len(train_loader)
            is_finished = False
            assert len(self.optimizers) == len(self.optimizer_closures) == 1
            assert (not any([isinstance(c, AbstractOptimizerClosure) for c in self.optimizer_closures])) or\
                   (self.accumulate_batches == 1), "Can not accumulate batches when optimizer(s) use closures"

            self._load_state_dict(load_state if isinstance(load_state, dict) else {})
            del load_state

            for i in range(epochs):
                self._log('Training, epoch %d, %d steps' % (self.get_method().current_epoch, num_steps))
                if is_finished:
                    self._log('The method finished, stopping early.')
                    break
                self._trigger_callbacks("on_epoch_start", self, self.get_method())

                # log regularizers, the learning rate, ...
                train_loader.get_dist_sampler().set_epoch(self.get_method().current_epoch)
                log_dict = self.get_method().on_epoch_start(log=False, is_last=(i == epochs-1))
                for clone in self.get_method_clones():
                    clone.get_method().on_epoch_start(log=False, is_last=(i == epochs-1))
                log_dict.update(self.get_optimizer_log_dict())
                self._log_dict(log_dict, do_print=False, sync=False)
                self._trigger_callbacks("on_train_epoch_start", self, self.get_method(), log_dict=log_dict)

                # train
                log_dict = self._train_steps_ddp(train_loader, num_steps, is_test_run)
                self.get_method().training_epoch_end([])
                self._log_dict(log_dict, do_print=True, sync=True)
                for clone in self.get_method_clones():
                    clone.on_training_epoch_end(self.get_method())
                self._trigger_callbacks("on_train_epoch_end", self, self.get_method(), log_dict=log_dict)

                # step the schedulers
                for scheduler in self.schedulers:
                    scheduler.step()

                # maybe eval and/or test
                if (eval_loader is not None) and (epochs - i <= eval_last or eval_last < 0 or is_finished):
                    eval_loader.get_dist_sampler().set_epoch(self.get_method().current_epoch)
                    self._trigger_callbacks("on_validation_epoch_start", self, self.get_method())
                    log_dict = self._eval_or_test_epoch_ddp(eval_loader, is_test_run=is_test_run, testing=False)
                    self._log_dict(log_dict, do_print=True, sync=True)
                    self._trigger_callbacks("on_validation_epoch_end", self, self.get_method(), log_dict=log_dict)
                if (test_loader is not None) and (epochs - i <= test_last or test_last < 0 or is_finished):
                    test_loader.get_dist_sampler().set_epoch(self.get_method().current_epoch)
                    self._trigger_callbacks("on_test_epoch_start", self, self.get_method())
                    log_dict = self._eval_or_test_epoch_ddp(test_loader, is_test_run=is_test_run, testing=True)
                    self._log_dict(log_dict, do_print=True, sync=True)
                    self._trigger_callbacks("on_test_epoch_end", self, self.get_method(), log_dict=log_dict)

                # plot/log accumulated metrics
                log_dict = {}
                for method, fmt in self.iterate_methods_on_device():
                    stats = method.get_accumulated_metric_stats(prefix=fmt % 'net')
                    stats = self._gather_tensor_dict(stats)
                    if self.rank == 0:
                        stats = method.eval_accumulated_metric_stats(
                            save_dir=self.get_metrics_save_dir('net'), stats=stats)
                        log_dict.update(stats)
                if (len(log_dict) > 0) and (self.rank == 0):
                    self._log('Accumulated stats, epoch %d' % self.get_method().current_epoch)
                    self._log_dict(log_dict, do_print=True, sync=False)

                # end epoch
                self.get_method().on_epoch_end(log=rank == 0)
                is_finished = self.get_method().is_finished()
                self._trigger_callbacks("on_epoch_end", self, self.get_method())

        self._trigger_callbacks("on_fit_end", self, self.get_method())
        self._trigger_callbacks("teardown", self, self.get_method(), "fit+test")
        for clone in self.get_method_clones():
            clone.stop()
        self._cleanup_ddp()

    def get_rank(self) -> int:
        return self.rank

    def is_test_run(self) -> bool:
        return self.is_test_run

    def _trigger_callbacks(self, callback_fun: str, *args, **kwargs):
        for c in self.callbacks:
            getattr(c, callback_fun)(*args, **kwargs)

    def get_save_dir(self) -> str:
        return self.save_dir

    def get_method_ddp(self) -> Ddp:
        return self.ddp_method

    def get_method_wrapper(self) -> MethodWrapperModule:
        return self.ddp_method.module

    def get_method(self) -> AbstractMethod:
        return self.get_method_wrapper().get_method()

    def get_method_clones(self) -> [AbstractMethodClone]:
        """ get the method clones """
        return self.method_clones

    def _log(self, msg: str):
        if self.rank == 0:
            self.logger.info(msg)

    def _setup_ddp(self, world_size: int, method: AbstractMethod, exp_logger: LightningLoggerBase, use_sync_bn: bool):
        # set environment vars and init process group
        os.environ['NCCL_BLOCKING_WAIT'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.rank)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(self.rank)
        dist.init_process_group('nccl', rank=self.rank, world_size=world_size, timeout=timedelta(minutes=5))
        assert self.rank == dist.get_rank()
        assert world_size == dist.get_world_size()

        # move model, initialize its clones, rank 0 logs everything
        has_unused_params = not method.uses_all_paths()
        method = self.mover.move_module(method)
        method.set_logger(exp_logger)
        for clone in self.method_clones:
            clone.init(method)
        if use_sync_bn:
            method = torch.nn.SyncBatchNorm.convert_sync_batchnorm(method)
        if self.rank == 0:
            method.log_hyperparams()
        self.ddp_method = Ddp(MethodWrapperModule(method),
                              device_ids=self.mover.get_indices(),
                              find_unused_parameters=has_unused_params)

        assert isinstance(self.ddp_method, Ddp)
        assert isinstance(self.get_method_ddp(), Ddp)
        assert isinstance(self.get_method_wrapper(), MethodWrapperModule)
        assert isinstance(self.get_method(), AbstractMethod)

    @classmethod
    def _cleanup_ddp(cls):
        dist.destroy_process_group()

    @classmethod
    def _sync_tensor_dict(cls, log_dict: {str: torch.Tensor}) -> {str: torch.Tensor}:
        """ return a copy of a {name: tensor} dict averaged over all processes """
        awaits, synced_dict, ws = [], {}, dist.get_world_size()
        log_dict_values, _ = LogResult.split_log_dict(log_dict)
        for k in log_dict_values.keys():
            synced_dict[k] = log_dict_values[k].clone().detach().div(ws)
            awaits.append(dist.reduce(synced_dict[k], op=dist.ReduceOp.SUM, dst=0, async_op=True))
        for a in awaits:
            a.wait()
        return synced_dict

    @classmethod
    def _gather_tensor_dict(cls, log_dict: {str: torch.Tensor}) -> {str: [torch.Tensor]}:
        """ gather a {name: tensor} over all processes to a {name: [tensor]} dict """
        awaits, synced_dict, ws = [], {}, dist.get_world_size()
        log_dict_values, _ = LogResult.split_log_dict(log_dict)
        for k in log_dict_values.keys():
            v = log_dict_values[k].clone().detach()
            synced_dict[k] = [torch.zeros_like(v) for _ in range(ws)]
            awaits.append(dist.all_gather(synced_dict[k], v, async_op=True))
        for a in awaits:
            a.wait()
        return synced_dict

    def _log_dict(self, log_dict: {str: torch.Tensor}, do_print=True, sync=True):
        if sync:
            log_dict = self._sync_tensor_dict(log_dict)
        if self.rank == 0:
            if do_print:
                rows = [(k, itemize(v)) for k, v in log_dict.items()]
                log_in_columns(self.logger, rows, min_widths=(60, 0), start_space=4)
            self.get_method().log_metrics(log_dict)

    def _next_batch(self, loader: CustomIterator, move=True) -> list:
        """ get the next batch of the loader, move all tensors to the gpu device """
        if move:
            return self.mover.move(loader.__next__())
        return loader.__next__()

    def _summarize_log_dicts(self, results: [LogResult]) -> LogResult:
        return self.get_method().summarize_outputs(results)

    def _train_steps_ddp(self, loader: CustomIterator, steps=1, is_test_run=False) -> dict:
        """ train 'steps' steps, return the method's log dict """
        self.get_method_wrapper().set_mode(train=True)
        results = []
        n = SimpleDDPTrainer.num_opt_steps(loader, dist.get_world_size(), is_test_run)
        for i in range(steps):
            opt, closure = self.optimizers[0], self.optimizer_closures[0]

            # maybe use closure
            if isinstance(closure, AbstractOptimizerClosure):
                c = closure.prepare(batch=self._next_batch(loader, move=False), batch_idx=i)
                self.get_method().optimizer_step(epoch=self.get_method().current_epoch, batch_idx=i,
                                                 optimizer=opt, optimizer_idx=0, optimizer_closure=c)
                results.append(c.get_result())
                for clone in self.get_method_clones():
                    clone.on_update(self.get_method())

            # default step
            else:
                result = self.get_method_ddp()(batch=self._next_batch(loader), batch_idx=i)
                result.backward()
                result.detach()
                results.append(result)

                self._acc_step += 1
                self._acc_step %= self.accumulate_batches
                if self._acc_step == 0:
                    self.get_method().optimizer_step(epoch=self.get_method().current_epoch, batch_idx=i,
                                                     optimizer=self.optimizers[0], optimizer_idx=0)
                    for clone in self.get_method_clones():
                        clone.on_update(self.get_method())

            for scheduler in self.schedulers:
                scheduler.step_samples(n=n)
        return self._summarize_log_dicts(results).get_log_info()

    def _eval_or_test_steps_ddp(self, loader: CustomIterator, steps=1, testing=False) -> dict:
        """ eval 'steps' steps for both model and model_ema, return the merged log dict """
        main_dict = {}
        with torch.no_grad():
            if loader is not None:
                # get usable methods, set them to eval mode
                methods_fmts = list(self.iterate_wrappers_on_device())
                results = []
                for (m, _) in methods_fmts:
                    m.set_mode(valid=not testing, test=testing)
                    results.append([])
                # iterate num batches, same batch for all methods
                for i in range(steps):
                    batch = self._next_batch(loader)
                    for j, (m, _) in enumerate(methods_fmts):
                        results[j].append(m(batch=batch, batch_idx=i))
                # add suffix to the string before the first /
                for r, (_, fmt) in zip(results, methods_fmts):
                    for k, v in self._summarize_log_dicts(r).get_log_info().items():
                        ks = k.split('/')
                        ks[0] = fmt % ks[0]
                        main_dict['/'.join(ks)] = v
            return main_dict

    def _eval_or_test_epoch_ddp(self, loader: CustomIterator, is_test_run=False, testing=False) -> dict:
        """ eval or test one epoch """
        name = 'Test' if testing else 'Eval'
        if loader is not None:
            num_steps = min([SimpleDDPTrainer.num_test_steps, len(loader)]) if is_test_run else len(loader)
            self._log('%s, epoch %d, %d steps' % (name, self.get_method().current_epoch, num_steps))
            return self._eval_or_test_steps_ddp(loader, steps=num_steps, testing=testing)
        return {}

    def _load_ddp(self, module: AbstractMethod, save_dir: str, prefer_ema=True):
        # not used in each process, no need to care for regular/ema model distinction
        file_names = ['checkpoint.tmp.pt', 'checkpoint.ema.tmp.pt']
        file_names = reversed(file_names) if prefer_ema else file_names
        for fn in file_names:
            file = SimpleDDPTrainer.checkpoint_file(save_dir, fn)
            if CheckpointCallback.load(file_path=file, pl_module=module):
                break

    def iterate_wrappers_on_device(self) -> Iterable[Tuple[AbstractMethod, str]]:
        """
        iterate the methods that are placed on the main device
        :return: pairs of (method, format string for log_dicts)
        """
        yield self.get_method_wrapper(), '%s'
        for clone in self.get_method_clones():
            if clone.is_on_same_device():
                yield clone, '%s/clones/%s' % ('%s', clone.get_name())

    def get_checkpoint_update_dict(self, *_) -> dict:
        """ get the internal state """
        # optional argument required for lightning
        return {'trainer_state': self._get_state_dict()}

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

    def get_optimizers(self) -> [Optimizer]:
        """ get optimizers """
        return self.optimizers

    def get_schedulers(self) -> [AbstractScheduler]:
        """ get schedulers """
        return self.schedulers


@Register.trainer()
class SimpleDDPTrainer(AbstractTrainer):
    """
    A simple trainer for a data-parallel training on multiple GPUs
    only supports eval/test as part of the training loop
    requires a checkpoint callback to save and recover weights
    """

    def __init__(self, method: AbstractMethod, args: Namespace, *_, **__):
        self.loader_train, self.loader_eval, self.loader_test = None, None, None
        self.optimizers, self.schedulers = [], []
        self.use_sync_bn = self._parsed_argument('use_sync_bn', args)
        super().__init__(method, args, *_, **__)

        if self.mover.get_num_devices() == 1:
            self.logger.warning('Using only one device for DDP training, consider using SimpleTrainer')

        self._state_to_load = dict()

        # set distributed sharing_strategy
        sharing_strategy = self._parsed_argument('sharing_strategy', args)
        if not sharing_strategy == 'default':
            mp.set_sharing_strategy(sharing_strategy)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        s = ['default'] + list(mp.get_all_sharing_strategies())
        return super().args_to_add(index) + [
            Argument('use_sync_bn', default='False', type=str, help='convert BatchNorm to SyncBatchNorm', is_bool=True),
            Argument('sharing_strategy', default='default', choices=s, type=str, help='how to share resources across processes'),
        ]

    def train_epochs(self, epochs=1, run_eval=True, run_test=True):
        """ train 'epochs' epochs, includes eval/test for the last n epochs """
        assert any([isinstance(c, CheckpointCallback) for c in self.callbacks]),\
            "DDP requires a checkpoint callback to recover weights later."
        self.mover.empty_cache()
        world_size = self.mover.get_num_devices()
        args = (world_size, self.method, self.method_clones,
                self.save_dir, self.mover, self.callbacks,
                self.exp_logger, epochs, self.eval_last, self.test_last, self._is_test_run,
                self.accumulate_batches, self.use_sync_bn, self._state_to_load)

        # create threads, join them manually, always try to clean up
        context = mp.spawn(SimpleDDPTrainerTrainEpochsImpl, args=args, nprocs=world_size, join=False)
        try:
            while not context.join():
                pass
            self.resource_logger.wakeup()
        except Exception as e:
            raise e
        finally:
            self.mover.empty_cache()
            self.resource_logger.stop()

        # done training, load weights in case something else happens
        CheckpointCallback.load_last_checkpoint(self.save_dir, self.method)

    def eval_epoch(self):
        """ eval one epoch """
        raise NotImplementedError

    def test_epoch(self):
        """ test one epoch """
        raise NotImplementedError

    def _load_state_dict(self, state: dict):
        """ load the internal state, buffer for the multiple spawned instances """
        self._state_to_load = state

    def get_optimizers(self) -> [Optimizer]:
        """ get optimizers """
        raise NotImplementedError

    def get_schedulers(self) -> [AbstractScheduler]:
        """ get schedulers """
        raise NotImplementedError
