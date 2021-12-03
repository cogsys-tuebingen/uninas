import torch.nn as nn
from uninas.training.devices.abstract import AbstractDevicesManager, AbstractDeviceMover, T
from uninas.register import Register


class CpuDeviceMover(AbstractDeviceMover):
    """
    handle data flow to cpu (mostly do nothing)
    """

    @property
    def name(self) -> str:
        return '%s()' % self.__class__.__name__

    def empty_cache(self):
        """
        empty the cache
        """
        pass

    def _synchronize(self, indices: [int]):
        """ make sure all operations are complete """
        pass

    def get_usage_dict(self, log_all=False) -> dict:
        """ return a dict that logs the usage of the device(s) """
        return {}

    def move_module(self, module: nn.Module) -> nn.Module:
        """ move module to the assigned devices """
        assert self.get_num_devices() == 1
        return module

    def _move(self, t: T) -> T:
        """ move (nested) tensors to the assigned devices """
        return t


@Register.devices_manager()
class CpuDevicesManager(AbstractDevicesManager):
    """
    manage allocation/de-allocation of one CPU device
    """
    _mover_cls = CpuDeviceMover

    def __init__(self, seed: int, is_deterministic: bool, num_devices: int):
        assert num_devices == 1
        super().__init__(seed, is_deterministic, num_devices)
