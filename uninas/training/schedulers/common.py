from torch import optim
from uninas.training.schedulers.abstract import AbstractScheduler
from uninas.utils.args import Argument, Namespace
from uninas.utils.misc import split
from uninas.register import Register


@Register.scheduler()
class ConstantScheduler(AbstractScheduler):
    """
    Keeps the same learning rate
    """

    @classmethod
    def scheduler_cls(cls, *_, **__):
        return None

    def __init__(self, args: Namespace, optimizer, max_epochs: int, index: int = None):
        super().__init__(args, optimizer, max_epochs, index)
        self.optimizer = optimizer

    def get_last_lr(self):
        return self.optimizer.get_lr()

    def get_lr(self):
        return self.optimizer.get_lr()

    def step(self, epoch=None):
        pass

    def step_samples(self, n=1):
        pass


@Register.scheduler()
class PiecewiseConstantScheduler(AbstractScheduler):
    """
    Multiplies the current learning rate with 'gamma' whenever a milestone epoch is reached
    """

    @classmethod
    def scheduler_cls(cls, optimizer=None, max_epochs=1, milestones='50, 100', gamma=0.1):
        milestones = split(milestones, cast_fun=int)
        return optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return [
            Argument('milestones', default="50, 100", type=str, help='List of epoch indices. Must be increasing'),
            Argument('gamma', default=0.1, type=float, help='Multiplicative factor of learning rate decay'),
        ] + super().args_to_add(index)


@Register.scheduler()
class ExponentialScheduler(AbstractScheduler):
    """
    Multiplies the current learning rate with 'gamma' after every epoch (or n steps)
    """

    @classmethod
    def scheduler_cls(cls, optimizer=None, max_epochs=1, gamma=0.95):
        return optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return [
            Argument('gamma', default=0.95, type=float, help='lr multiplier after each epoch'),
        ] + super().args_to_add(index)


@Register.scheduler()
class CosineScheduler(AbstractScheduler):
    """
    Cosine schedule
    """

    @classmethod
    def scheduler_cls(cls, optimizer=None, max_epochs=1, eta_min=0.0):
        return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=eta_min, T_max=max_epochs)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return [
            Argument('eta_min', default=0.0, type=float, help='lr at the end of the schedule'),
        ] + super().args_to_add(index)
