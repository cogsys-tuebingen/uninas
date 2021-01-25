import os
import torch
from datetime import timedelta
from typing import Union
from torch.nn.parallel import DistributedDataParallel as Ddp
from torch.optim.optimizer import Optimizer
import torch.multiprocessing as mp
import torch.distributed as dist
from pytorch_lightning.accelerators.ddp_accelerator import DDPAccelerator
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from uninas.methods.abstract import AbstractMethod
from uninas.training.trainer.abstract import AbstractTrainerFunctions
from uninas.training.trainer.abstract2 import AbstractTrainer
from uninas.training.schedulers.abstract import AbstractScheduler
from uninas.training.result import EvalLogResult, TrainLogResult
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.training.callbacks.abstract import AbstractCallback
from uninas.training.callbacks.checkpoint import CheckpointCallback
from uninas.utils.loggers.exp import LightningLoggerBase
from uninas.utils.args import Argument, Namespace
from uninas.utils.torch.ema import ModelEMA
from uninas.register import Register


class SimpleDDPTrainerTrainEpochsImpl(AbstractTrainerFunctions):
    """
    Train until epoch, ddp
    """

    def __init__(self, rank: int, world_size: int, method: AbstractMethod, save_dir: str, mover: AbstractDeviceMover,
                 callbacks: [AbstractCallback], exp_logger: LightningLoggerBase, epochs=1, eval_last=2, test_last=2,
                 is_test_run=False, ema_decay=0.9999, ema_device='same', use_sync_bn=False, load_state: dict = None):
        super().__init__()
        self.rank = rank
        self.mover = mover.get_device_subset([rank])
        self.logger = SimpleDDPTrainer.get_logger(None, is_test_run, save_dir, suffix=str(rank))

        self.ddp_method, self.method_ema =\
            self._setup_ddp(world_size, method, exp_logger, ema_decay, ema_device, use_sync_bn)
        self.optimizers, self.schedulers = self.get_method().configure_optimizers()
        self.callbacks = callbacks
        for c in self.callbacks:
            c.setup(self, self.get_method(), "fit+test")

        if epochs > 0:
            train_loader = self.get_method().data_set.train_loader(dist=True)
            eval_loader = self.get_method().data_set.valid_loader(dist=True) if eval_last != 0 else None
            test_loader = self.get_method().data_set.test_loader(dist=True) if test_last != 0 else None

            num_steps = min([SimpleDDPTrainer.num_test_steps, len(train_loader)]) if is_test_run else len(train_loader)
            is_finished = False
            assert len(self.optimizers) == 1

            self._load_state_dict(load_state if isinstance(load_state, dict) else {})
            del load_state

            for i in range(epochs):
                self._log('Training, epoch %d, %d steps' % (self.get_method().current_epoch, num_steps))
                if is_finished:
                    self._log('The method finished, stopping early.')
                    break

                # log regularizers, the learning rate, ...
                train_loader.sampler.set_epoch(self.get_method().current_epoch)
                log_dict = self.get_method().on_epoch_start(log=False)
                log_dict.update(self.get_optimizer_log_dict())
                self._log_dict(log_dict, do_print=False, sync=False, callback_fun='on_train_epoch_start')

                # train
                log_dict = self._train_steps_ddp(train_loader, num_steps, is_test_run)
                self.get_method().training_epoch_end([])
                if self.method_ema is not None:
                    assert isinstance(self.get_method(), AbstractMethod)
                    self.method_ema.update(self.get_method())
                self._log_dict(log_dict, do_print=True, sync=True, callback_fun='on_train_epoch_end')

                # step the schedulers
                for scheduler in self.schedulers:
                    scheduler.step()

                # maybe eval and/or test
                if (eval_loader is not None) and (epochs - i <= eval_last or eval_last < 0 or is_finished):
                    e = self.get_method().current_epoch
                    eval_loader.sampler.set_epoch(e)
                    log_dict = self._eval_or_test_epoch_ddp(eval_loader, is_test_run=is_test_run, testing=False)
                    self._log_dict(log_dict, do_print=True, sync=True, epoch=e, callback_fun='on_validation_epoch_end')
                if (test_loader is not None) and (epochs - i <= test_last or test_last < 0 or is_finished):
                    e = self.get_method().current_epoch
                    test_loader.sampler.set_epoch(e)
                    log_dict = self._eval_or_test_epoch_ddp(test_loader, is_test_run=is_test_run, testing=True)
                    self._log_dict(log_dict, do_print=True, sync=True, epoch=e, callback_fun='on_test_epoch_end')

                self.get_method().on_epoch_end(log=rank == 0)
                is_finished = self.get_method().is_finished()

        for c in self.callbacks:
            c.teardown(self, self.get_method(), "fit+test")
        self._cleanup_ddp()

    def get_method(self) -> AbstractMethod:
        m = self.ddp_method.module
        assert isinstance(m, AbstractMethod)
        return m

    def _log(self, msg: str):
        if self.rank == 0:
            self.logger.info(msg)

    def _setup_ddp(self, world_size: int, method: AbstractMethod, exp_logger: LightningLoggerBase,
                   ema_decay: float, ema_device: str, use_sync_bn: bool) -> (Ddp, ModelEMA):
        os.environ['NCCL_BLOCKING_WAIT'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(self.rank)
        dist.init_process_group('nccl', rank=self.rank, world_size=world_size, timeout=timedelta(minutes=5))

        accelerator = DDPAccelerator(None, ddp_plugin=DDPPlugin())

        if use_sync_bn:
            method = accelerator.configure_sync_batchnorm(method)

        method = self.mover.move_module(method)
        method_ema = ModelEMA.maybe_init(self.logger, method, ema_decay, ema_device)

        method.logger = exp_logger
        if method_ema is not None:
            method_ema.module.logger = method.logger
        if self.rank == 0:
            method.log_hyperparams()
        assert self.rank == dist.get_rank()
        assert world_size == dist.get_world_size()

        return accelerator.configure_ddp(method, device_ids=[self.rank]), method_ema

    @classmethod
    def _cleanup_ddp(cls):
        dist.destroy_process_group()

    @classmethod
    def _sync_log_dict(cls, log_dict: dict) -> dict:
        """ return a copy of a {name: tensor} dict averaged over all processes """
        awaits, synced_dict, ws = [], {}, dist.get_world_size()
        for k in log_dict.keys():
            synced_dict[k] = log_dict[k].clone().detach().div(ws)
            awaits.append(dist.reduce(synced_dict[k], op=dist.ReduceOp.SUM, dst=0, async_op=True))
        for a in awaits:
            a.wait()
        return synced_dict

    def _log_dict(self, log_dict: {str: torch.Tensor}, do_print=True, sync=True, epoch=None, callback_fun: str = None):
        method = self.choose_method(self.ddp_method, self.method_ema, prefer_ema=False)
        epoch = epoch if epoch is not None else method.current_epoch
        if sync:
            log_dict = self._sync_log_dict(log_dict)
        if self.rank == 0:
            if do_print:
                for k, v in log_dict.items():
                    self.logger.info('    {:<30}{}'.format(k, v.item()))
            method.logger.log_metrics(log_dict, epoch)
            # callbacks
            if isinstance(callback_fun, str):
                for c in self.callbacks:
                    getattr(c, callback_fun)(self, method, self.method_ema, log_dict=log_dict)

    def _next_batch(self, loader) -> list:
        """ get the next batch of the loader, move all tensors to the gpu device """
        return self.mover.move(loader.__next__())

    def _summarize_log_dicts(self, results: [Union[EvalLogResult, TrainLogResult]]) -> EvalLogResult:
        return self.get_method().validation_epoch_end(results)

    def _train_steps_ddp(self, loader, steps=1, is_test_run=False) -> dict:
        """ train 'steps' steps, return the method's log dict """
        self.ddp_method.train()
        self.get_method().testing = False
        results = []
        n = SimpleDDPTrainer.num_opt_steps(loader, dist.get_world_size(), is_test_run)
        for i in range(steps):
            result = self.ddp_method(batch=self._next_batch(loader), batch_idx=i)
            result.minimize.backward()
            result.detach()
            results.append(result)
            self.get_method().optimizer_step(epoch=self.get_method().current_epoch, batch_idx=i,
                                             optimizer=self.optimizers[0], optimizer_idx=0)
            for scheduler in self.schedulers:
                scheduler.step_samples(n=n)
        return self._summarize_log_dicts(results).get_log_info()

    def _eval_or_test_steps_ddp(self, loader, steps=1, testing=False) -> dict:
        """ eval 'steps' steps for both model and model_ema, return the merged log dict """
        main_dict = {}
        with torch.no_grad():
            if loader is not None:
                # get usable methods, set them to eval mode
                methods_fmts = list(self.iterate_usable_methods(self.ddp_method, self.method_ema))
                results = []
                for (m, _) in methods_fmts:
                    m.eval()
                    m.module.testing = testing
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

    def _eval_or_test_epoch_ddp(self, loader, is_test_run=False, testing=False) -> dict:
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
    """

    def __init__(self, method: AbstractMethod, args: Namespace, *_, **__):
        self.loader_train, self.loader_eval, self.loader_test = None, None, None
        self.optimizers, self.schedulers = [], []
        self.use_sync_bn = self._parsed_argument('use_sync_bn', args)
        super().__init__(method, args, *_, **__)

        if self.mover.get_num_devices() == 1:
            self.logger.warning('Using only one device for DDP training, consider using SimpleTrainer')

        self._state_to_load = dict()

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('use_sync_bn', default='False', type=str, help='convert BatchNorm to SyncBatchNorm', is_bool=True),
        ]

    def train_epochs(self, epochs=1, run_eval=True, run_test=True):
        """ train 'epochs' epochs, includes eval/test for the last n epochs """
        assert len(self.callbacks) > 0, "DDP requires a checkpoint callback to recover weights later"
        self.mover.empty_cache()
        args = (self.mover.get_num_devices(), self.method, self.save_dir, self.mover, self.callbacks,
                self.exp_logger, epochs, self.eval_last, self.test_last, self.is_test_run,
                self.ema_decay, self.ema_device, self.use_sync_bn, self._state_to_load)
        # create threads, join them, always try to clean up
        context = mp.spawn(SimpleDDPTrainerTrainEpochsImpl, args=args, nprocs=self.mover.get_num_devices(), join=False)
        try:
            while not context.join():
                pass
            self.resource_logger.wakeup()
        except Exception as e:
            raise e
        finally:
            self.mover.empty_cache()
            self.resource_logger.stop()
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
