import typing
import torch
import torch.nn as nn
from uninas.utils.args import ArgsInterface, Argument, Namespace


T = typing.TypeVar('T')


class AbstractDeviceMover:
    """
    handle data flow to specific devices
    """

    def __init__(self, handler, indices: [int]):
        self.handler = handler
        self.indices = indices
        self._original_indices = indices.copy()
        self._is_alive = True

    @property
    def name(self) -> str:
        return '%s(%s)' % (self.__class__.__name__, str(self.indices))

    def get_indices(self) -> [int]:
        return self.indices

    def get_device_subset(self, indices: [int]) -> 'AbstractDeviceMover':
        """ remove indices/devices from this mover, return a new one that has them """
        for i in indices:
            assert i in self.indices, "Can not give a device that is not owned"
            self.indices.remove(i)
        return self.__class__(self.handler, indices)

    def get_num_devices(self) -> int:
        """ number of devices """
        return len(self.indices)

    def set_rank(self):
        """ set the rank (in distributed training) to all available remaining devices """
        pass

    def deallocate(self):
        """ deallocate the devices so that the handler can allocate them again, irreversible """
        if self._is_alive:
            self.handler.deallocate_devices(self)
            self._is_alive = False

    def move(self, t: T) -> T:
        """ move the tensor(s) to the assigned devices """
        assert self._is_alive, "Can not move to device after de-allocating it"
        return self._move(t)

    def empty_cache(self):
        """
        empty the cache
        """
        raise NotImplementedError

    def synchronize(self, original=True):
        """
        make sure all operations are complete
        if original is True, consider all devices (in case this is called on a subset of a previous Mover)
        """
        self._synchronize(self._original_indices if original else self.indices)

    def _synchronize(self, indices: [int]):
        """ make sure all operations are complete """
        raise NotImplementedError

    def get_usage_dict(self, log_all=False) -> dict:
        """ return a dict that logs the usage of the device(s) """
        raise NotImplementedError

    def move_module(self, module: nn.Module) -> nn.Module:
        """ move module to the assigned devices """
        raise NotImplementedError

    def _move(self, t: T) -> T:
        """ move (nested) tensors to the assigned devices """
        raise NotImplementedError


class AbstractDevicesManager(ArgsInterface):
    """
    manage allocation/de-allocation of specific devices
    """
    _mover_cls = AbstractDeviceMover

    def __init__(self, seed: int, is_deterministic: bool, num_devices: int):
        super().__init__()
        assert num_devices > 0, "Must have more than zero devices!"
        self.num_devices = num_devices
        self.all_devices = list(range(num_devices))
        self.free_devices = self.all_devices.copy()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_num_devices(self) -> int:
        """ number of total devices """
        return self.num_devices

    def get_num_free(self) -> int:
        """ number of free devices """
        return len(self.free_devices)

    def allocate_devices(self, num: int) -> AbstractDeviceMover:
        """
        allocate 'num' devices and get the indices,
        allocate all if 'num' is smaller than zero
        """
        assert self.get_num_free() >= num, "Can not allocate %d devices, only %d available"\
                                           % (num, len(self.free_devices))
        if num < 0:
            num = self.get_num_free()
        return self._mover_cls(self, [self.free_devices.pop() for _ in range(num)])

    def deallocate_devices(self, mover: AbstractDeviceMover):
        """
        de-allocate (free) the devices for further use
        """
        n = len(self.free_devices)
        self.free_devices.extend(mover.indices)
        assert len(self.free_devices) == n + len(mover.indices), "de-allocated devices that were already free"

    @classmethod
    def get_device(cls, module: nn.Module) -> torch.device:
        """ figure out which device the parameters of a model are on """
        for n in module.parameters(recurse=True):
            return n.device
        return torch.device('cpu')

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('num_devices', default=1, type=int, help='number of available devices'),
        ]

    @classmethod
    def from_args(cls, seed: int, is_deterministic: bool, args: Namespace, index: int = None) -> 'AbstractDevicesManager':
        parsed = cls._all_parsed_arguments(args, index=index)
        return cls(seed, is_deterministic, **parsed)
