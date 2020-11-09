import time
import torch
import torch.nn as nn
import typing
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.utils.args import ArgsInterface, Argument, Namespace
from uninas.utils.shape import Shape, ShapeList
from uninas.register import Register

ShapeOrList = typing.Union[Shape, ShapeList]
TensorOrList = typing.Union[torch.Tensor, typing.List[torch.Tensor]]


class AbstractProfileFunction(ArgsInterface):

    @classmethod
    def from_args(cls, args: Namespace, index=None):
        return cls(**cls._all_parsed_arguments(args, index=index))

    def __init__(self, **__):
        super().__init__()

    def profile(self, module: nn.Module, shape_in: Shape, mover: AbstractDeviceMover, batch_size: int) -> float:
        raise NotImplementedError


@Register.profile_function()
class LatencyProfileFunction(AbstractProfileFunction):

    def __init__(self, num_warmup=10, num_measure=10):
        super().__init__()
        self.num_warmup = num_warmup
        self.num_measure = num_measure

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('num_warmup', default=10, type=int, help='warmup forward passes to stabilize the results'),
            Argument('num_measure', default=10, type=int, help='average over this many forward passes'),
        ]

    def profile(self, module: nn.Module, shape_in: ShapeOrList, mover: AbstractDeviceMover, batch_size: int) -> float:
        with torch.no_grad():
            inputs_ = mover.move(shape_in.random_tensor(batch_size=batch_size))
            for _ in range(self.num_warmup):
                module(inputs_)
            mover.synchronize(original=True)
            t0 = time.time()
            for _ in range(self.num_measure):
                module(inputs_)
                mover.synchronize(original=True)
            return (time.time() - t0) / self.num_measure
