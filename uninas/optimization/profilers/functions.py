"""
functions to profile latency/params/macs/...

should investigate in the future:
https://pytorch.org/docs/1.8.0/benchmark_utils.html?highlight=benchmark#
https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_nvtx
"""

import time
import torch
import torch.nn as nn
import typing
import torchprofile
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.training.devices.cpu import CpuDeviceMover
from uninas.utils.args import ArgsInterface, Argument, Namespace
from uninas.utils.shape import Shape, ShapeList
from uninas.register import Register

ShapeOrList = typing.Union[Shape, ShapeList]
TensorOrList = typing.Union[torch.Tensor, typing.List[torch.Tensor]]


class AbstractProfileFunction(ArgsInterface):

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> 'AbstractProfileFunction':
        return cls(**cls._all_parsed_arguments(args, index=index))

    def __init__(self, **__):
        super().__init__()

    def profile(self, module: nn.Module, shape_in: Shape, mover: AbstractDeviceMover, batch_size: int) -> float:
        raise NotImplementedError


class AbstractLatencyProfileFunction(AbstractProfileFunction):
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
        raise NotImplementedError


@Register.profile_function()
class LatencyTimeProfileFunction(AbstractLatencyProfileFunction):
    """
    Measure model latency with python time.time() queries in seconds,
    works for any device, but is subject to time spent in python code
    """

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


@Register.profile_function()
class LatencyTorchProfileFunction(AbstractLatencyProfileFunction):
    """
    Measure model latency with torch autograd profiler in seconds,
    currently works only for CPU, but is not subject to time spent in python code
    """

    def profile(self, module: nn.Module, shape_in: ShapeOrList, mover: AbstractDeviceMover, batch_size: int) -> float:
        assert isinstance(mover, CpuDeviceMover), "The torch profiler can only measure CPU latencies"
        with torch.no_grad():
            inputs_ = mover.move(shape_in.random_tensor(batch_size=batch_size))
            for _ in range(self.num_warmup):
                module(inputs_)
            mover.synchronize(original=True)
            with torch.autograd.profiler.profile() as profiler:
                for _ in range(self.num_measure):
                    module(inputs_)
                    mover.synchronize(original=True)
            return (profiler.self_cpu_time_total / self.num_measure) / 1e6  # cast time to seconds


@Register.profile_function()
class MacsProfileFunction(AbstractProfileFunction):
    """
    Count the number of MACs of all used modules
    """

    def profile(self, module: nn.Module, shape_in: ShapeOrList, mover: AbstractDeviceMover, batch_size: int) -> float:
        with torch.no_grad():
            inputs_ = mover.move(shape_in.random_tensor(batch_size=batch_size))
            return torchprofile.profile_macs(module, args=inputs_) // batch_size


@Register.profile_function()
class ParamsProfileFunction(AbstractProfileFunction):
    """
    Count the number of parameters of all used modules
    not accurate for mixed modules or separate modules with shared parameters
    """

    def profile(self, module: nn.Module, shape_in: ShapeOrList, mover: AbstractDeviceMover, batch_size: int) -> float:
        return sum([m.numel() for m in module.parameters(recurse=True)])
