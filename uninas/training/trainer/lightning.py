import os
import types
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer

from uninas.methods.abstract_method import AbstractMethod
from uninas.training.trainer.abstract2 import AbstractTrainer
from uninas.training.schedulers.abstract import AbstractScheduler
from uninas.utils.args import MetaArgument, Argument, Namespace
from uninas.register import Register


@Register.trainer()
class LightningTrainer(AbstractTrainer):
    """
    THIS TRAINER IS CURRENTLY NOT UP TO DATE

    Wrapping the Pytorch-Lightning trainer

    this trainer is missing somewhat special requirements:
        - stepping schedulers each n steps, not only each epoch
        - run eval/test the last n epochs
        - handling variable-length datasets (i.e. MixedLoader)
        - train/eval/test for exactly n steps
    and the trainer ignores the cls_device, it will handle that part itself

    furthermore, in my own experiments, the simple trainer implementations are significantly faster
    (in <0.8 versions however, that might have changed)
    """

    can_use_ema = False
    can_eval_n = False

    def __init__(self, method: AbstractMethod, args: Namespace, *_, **__):
        super().__init__(method, args, *_, **__)
        self.weights_dir = '%sweights/' % self.save_dir
        os.makedirs(self.weights_dir, exist_ok=True)

        cudnn, cudnn_b, cudnn_d = self._parsed_arguments(['cudnn', 'cudnn_benchmark', 'cudnn_deterministic'], args)
        nodes, gpus, backend = self._parsed_arguments(['distributed_nodes', 'distributed_gpus', 'distributed_backend'], args)

        # in test runs, have only 'num_test_steps' steps per epoch
        train_pc, val_pc, test_pc = 1.0, 1.0, 1.0
        if self._is_test_run:
            train_pc = min([1, self.num_test_steps / len(self.method.train_dataloader() or range(self.num_test_steps))])
            val_pc = min([1, self.num_test_steps / len(self.method.val_dataloader() or range(self.num_test_steps))])
            test_pc = min([1, self.num_test_steps / len(self.method.test_dataloader() or range(self.num_test_steps))])

        cf = self.checkpoint_file(self.save_dir)
        self.trainer = pl.Trainer(logger=self.exp_logger,
                                  profiler=self.mover.get_num_devices() == 1,
                                  # row_log_interval=10,
                                  weights_summary=None,
                                  checkpoint_callback=False,
                                  callbacks=self.callbacks,
                                  resume_from_checkpoint=cf if os.path.isfile(cf) else None,
                                  weights_save_path=self.weights_dir,
                                  max_epochs=999999999,
                                  check_val_every_n_epoch=999999999,
                                  # show_progress_bar=False,
                                  # gradient_clip_val=self.cg_v,  # maybe find callback, replace it?
                                  accumulate_grad_batches=self.accumulate_batches,
                                  early_stop_callback=False,
                                  gpus=gpus,
                                  auto_select_gpus=True,
                                  num_nodes=nodes,
                                  benchmark=cudnn_b,
                                  deterministic=cudnn_d and not cudnn_b,
                                  distributed_backend=backend,
                                  replace_sampler_ddp=True,
                                  train_percent_check=train_pc,
                                  val_percent_check=val_pc,
                                  test_percent_check=test_pc,
                                  log_gpu_memory=None)

        # compatibility
        self.trainer.get_checkpoint_update_dict = types.MethodType(self.get_checkpoint_update_dict, self.trainer)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            # distributing the network + data
            Argument('distributed_nodes', default=1, type=int, help='distribute training to this many nodes'),
            Argument('distributed_gpus', default=1, type=int, help='distribute training to this many gpus per node'),
            Argument('distributed_method', default='dp', type=str, help='how to distribute models across gpus/nodes',
                     choices=['dp', 'ddp', 'ddp2', 'horovod']),

            # cudnn settings
            Argument('cudnn', default='True', type=str, help='try using cudnn', is_bool=True),
            Argument('cudnn_benchmark', default='True', type=str, help='use cudnn benchmark', is_bool=True),
            Argument('cudnn_deterministic', default='False', type=str, help='use cudnn deterministic', is_bool=True),
        ]

    @classmethod
    def meta_args_to_add(cls, has_log_dict=True) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add(has_log_dict=False)

    def train_epochs(self, epochs=1, run_eval=True, run_test=True):
        """ train 'epochs' epochs """
        self.trainer.max_epochs = self.method.current_epoch + epochs
        self.trainer.fit(self.method)
        if run_eval:
            self.eval_epoch()
        if run_test:
            self.test_epoch()

    def eval_epoch(self):
        """ eval one epoch """
        try:
            self.trainer.run_evaluation()
        except Exception as e:
            self.logger.error('Exception when trying to eval.', exc_info=e)

    def test_epoch(self) -> 'LightningTrainer':
        """ test one epoch """
        self.trainer.test(self.method)
        return self

    def get_optimizers(self) -> [Optimizer]:
        """ get optimizers """
        raise NotImplementedError

    def get_schedulers(self) -> [AbstractScheduler]:
        """ get schedulers """
        raise NotImplementedError
