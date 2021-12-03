import time
import torch.nn as nn
from uninas.training.devices.abstract import AbstractDevicesManager, AbstractDeviceMover, T
from uninas.register import Register


class TestCpuDeviceMover(AbstractDeviceMover):
    """
    handle data flow to cpu (mostly do nothing), print what happens
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t0 = 0

    def _print(self, msg: str):
        print('%s: td=%d, idx=%s: %s' % (self.__class__.__name__, int(time.time() - self.t0), str(self.indices), msg))

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
        self._print("synchronize")
        pass

    def get_usage_dict(self, log_all=False) -> dict:
        """ return a dict that logs the usage of the device(s) """
        return {}

    def move_module(self, module: nn.Module) -> nn.Module:
        """ move module to the assigned devices """
        assert self.get_num_devices() == 1
        self.t0 = time.time()
        self._print("moving module, starting timer")
        return module

    def _move(self, t: T) -> T:
        """ move (nested) tensors to the assigned devices """
        self._print("moving data (%s)" % type(t))
        return t


@Register.devices_manager()
class TestCpuDevicesManager(AbstractDevicesManager):
    """
    manage allocation/de-allocation of one CPU device, which may be used multiple times for debug purposes
    """
    _mover_cls = TestCpuDeviceMover

    def __init__(self, seed: int, is_deterministic: bool, num_devices: int):
        super().__init__(seed, is_deterministic, num_devices)
