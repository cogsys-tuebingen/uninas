import torch
from torch.optim.sgd import SGD
from uninas.utils.args import Argument
from uninas.training.optimizers.abstract import WrappedOptimizer
from uninas.register import Register


class SGDDebug(SGD):
    """
    Classic SGD without momentum,

    added print statements to count how many parameters have:
    (no gradient, a non-zero gradient, a zero-gradient, do not require a gradient)
    """

    def __init__(self, params, lr, **__):
        super().__init__(params, lr)

    @classmethod
    def print(cls, *args, **kwargs):
        print(cls.__name__, *args, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            counts = [0, 0, 0, 0]
            for p in group['params']:
                if not p.requires_grad:
                    counts[3] += 1
                    continue
                if p.grad is None:
                    counts[0] += 1
                    continue
                d_p = p.grad

                if d_p.abs().sum() > 0:
                    counts[1] += 1
                else:
                    counts[2] += 1

                p.add_(d_p, alpha=-group['lr'])
            self.print('gradients:    None={:<8} non-zero={:<8} zero={:<8} not-req={:<8}'.format(*counts))
        return loss

    def zero_grad(self, *args, **kwargs) -> None:
        self.print('zero grad')
        super().zero_grad(*args, **kwargs)


@Register.optimizer()
class DebugOptimizer(WrappedOptimizer):
    optimizer_cls = SGDDebug

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return [
            Argument('lr', default=0.01, type=float, help='learning rate'),
        ] + super().args_to_add(index)
