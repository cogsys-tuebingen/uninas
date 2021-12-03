from torch import optim
from uninas.utils.args import ArgsInterface, Argument, Namespace


class AbstractScheduler(ArgsInterface, optim.lr_scheduler._LRScheduler):
    """
    base class for schedulers
    also wraps them to provide consistent warmup/cooldown and step/epoch update handling
    """

    def __init__(self, args: Namespace, optimizer, max_epochs: int, index: int = None):
        super().__init__()
        self.optimizer = optimizer
        all_kwargs = self._all_parsed_arguments(args=args, index=index)
        self._all_kwargs = all_kwargs.copy()
        self._each_samples = all_kwargs.pop('each_samples')
        self._cooldown_epochs = all_kwargs.pop('cooldown_epochs')
        self._warmup_epochs = all_kwargs.pop('warmup_epochs')
        self._warmup_lr = all_kwargs.pop('warmup_lr')
        self._warmup_lr_final = [group['lr'] for group in self.optimizer.param_groups]
        self._fake_num_epochs = all_kwargs.pop('fake_num_epochs')
        self._acc_samples = 0
        self._step_samples = self._each_samples > 0
        self._last_epoch = 0

        if self._fake_num_epochs < 0:
            # default schedule, warmup-train-cooldown
            num_epochs = max_epochs - self._warmup_epochs - self._cooldown_epochs
        elif self._fake_num_epochs < max_epochs:
            # keep original cooldown on num epochs, extend schedule with same lr
            num_epochs = self._fake_num_epochs - self._warmup_epochs - self._cooldown_epochs
            self._cooldown_epochs += (max_epochs - self._fake_num_epochs)
        else:
            # push cooldown further behind
            num_epochs = self._fake_num_epochs - self._warmup_epochs - self._cooldown_epochs
            self._cooldown_epochs = max([self._cooldown_epochs - (self._fake_num_epochs - max_epochs), 0])
        self._start_regular = self._warmup_epochs
        self._start_cooldown = self._warmup_epochs + num_epochs
        assert self._warmup_epochs < max_epochs
        assert num_epochs > 0, "have 0 epochs that are not warmup/cooldown"
        self._warmup_step()
        self.scheduler = self.scheduler_cls(self.optimizer, num_epochs, **all_kwargs)
        # print(self._start_regular, self._start_cooldown)

    @classmethod
    def from_args(cls, args: Namespace, optimizer, max_epochs: int, index: int = None):
        return cls(args, optimizer, max_epochs, index)

    def _str_dict(self) -> dict:
        return self._all_kwargs

    def state_dict(self) -> dict:
        return {
            'scheduler': None if self.scheduler is None else self.scheduler.state_dict(),
            '_all_kwargs': self._all_kwargs,
            '_each_samples': self._each_samples,
            '_acc_samples': self._acc_samples,
            '_last_epoch': self._last_epoch,
            '_cooldown_epochs': self._cooldown_epochs,
            '_warmup_epochs': self._warmup_epochs,
            '_warmup_lr': self._warmup_lr,
            '_warmup_lr_final': self._warmup_lr_final,
            '_fake_num_epochs': self._fake_num_epochs,
        }

    def load_state_dict(self, state_dict: dict):
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler'])
        self._all_kwargs = state_dict['_all_kwargs']
        self._each_samples = state_dict['_each_samples']
        self._acc_samples = state_dict['_acc_samples']
        self._last_epoch = state_dict['_last_epoch']
        self._cooldown_epochs = state_dict['_cooldown_epochs']
        self._warmup_epochs = state_dict['_warmup_epochs']
        self._warmup_lr = state_dict['_warmup_lr']
        self._warmup_lr_final = state_dict['_warmup_lr_final']
        self._fake_num_epochs = state_dict['_fake_num_epochs']
        self._step_samples = self._each_samples > 0

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def get_lr(self):
        return self.scheduler.get_lr()

    def is_in_warmup_phase(self) -> bool:
        return self._last_epoch <= self._start_regular

    def is_in_regular_phase(self) -> bool:
        return self._start_regular <= self._last_epoch <= self._start_cooldown

    def is_in_cooldown_phase(self) -> bool:
        return self._start_cooldown < self._last_epoch

    def _warmup_step(self):
        if self.is_in_warmup_phase():
            de = (self._last_epoch + 1) / (self._warmup_epochs + 1)
            lrs = [self._warmup_lr + (lr-self._warmup_lr) * de for lr in self._warmup_lr_final]
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr

    def step(self, epoch=None):
        """ at the end of each epoch """
        self._last_epoch += 1
        if self.is_in_warmup_phase():
            self._warmup_step()
        elif self.is_in_cooldown_phase():
            pass
        elif not self._step_samples:
            self.scheduler.step(epoch=epoch)

    def step_samples(self, n=1):
        """ after each optimizer step """
        if self._step_samples:
            if self.is_in_regular_phase():
                self._acc_samples += n
                while self._acc_samples >= self._each_samples:
                    self._acc_samples -= self._each_samples
                    self.scheduler.step()

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('each_samples', default=-1, type=int, help='step the scheduler each n samples instead of each epoch, if >0 (does not account for accumulated gradients)'),
            Argument('cooldown_epochs', default=0, type=int, help='remain at the final lr at the end'),
            Argument('warmup_epochs', default=0, type=int, help='linearly increase the lr to the initial lr over this many epochs, start the regular scheduler afterwards'),
            Argument('warmup_lr', default=0.0, type=float, help='initial lr when using warmup, the first value is skipped'),
            Argument('fake_num_epochs', default=-1, type=int, help='set up the schedule for a different number of epochs, if > 0'),
        ]

    @classmethod
    def scheduler_cls(cls, optimizer=None, max_epochs=1, **parsed_args):
        """ get a torch LR scheduler """
        raise NotImplementedError
