import GPUtil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel.scatter_gather import scatter
from uninas.training.devices.abstract import AbstractDevicesManager, AbstractDeviceMover, T
from uninas.utils.args import Argument
from uninas.register import Register


class CudaDeviceMover(AbstractDeviceMover):
    """
    handle data flow to specific CUDA devices
    """

    def set_rank(self):
        """ set the rank (in distributed training) to all available remaining devices """
        assert self.get_num_devices() == 1
        torch.cuda.set_device(self.get_indices()[0])

    def empty_cache(self):
        """
        empty the cache
        """
        torch.cuda.empty_cache()

    def _synchronize(self, indices: [int]):
        """ make sure all operations are complete """
        for i in indices:
            torch.cuda.synchronize(i)

    def get_usage_dict(self, log_all=False) -> dict:
        """ return a dict that logs the usage of the device(s) """
        dct = {}
        for gpu in GPUtil.getGPUs():
            if gpu.id in self._original_indices:
                if log_all:
                    dct['cuda/%d/%s' % (gpu.id, 'memoryTotal')] = gpu.memoryTotal
                    dct['cuda/%d/%s' % (gpu.id, 'memoryUsed')] = gpu.memoryUsed
                    dct['cuda/%d/%s' % (gpu.id, 'memoryFree')] = gpu.memoryFree
                dct['cuda/%d/%s' % (gpu.id, 'memoryUtil')] = gpu.memoryUtil
        return dct

    def move_module(self, module: nn.Module) -> nn.Module:
        """ move module to the assigned devices """
        assert self.get_num_devices() == 1
        return module.cuda(device=self.indices[0])

    def _move(self, t: T) -> T:
        """ move (nested) tensors to the assigned devices """
        return scatter(t, target_gpus=self.indices)[0]


@Register.devices_manager()
class CudaDevicesManager(AbstractDevicesManager):
    """
    manage allocation/de-allocation of CUDA devices
    """
    _mover_cls = CudaDeviceMover

    def __init__(self, seed: int, is_deterministic: bool, num_devices: int,
                 use_cudnn: bool, use_cudnn_benchmark: bool):
        if num_devices < 0:
            num_devices = torch.cuda.device_count()
        super().__init__(seed, is_deterministic, num_devices)
        assert torch.cuda.device_count() >= num_devices,\
            "Only %d devices available on the system, requesting %d" % (torch.cuda.device_count(), num_devices)
        if num_devices > 0:
            torch.cuda.manual_seed_all(seed)
            cudnn.set_flags(_enabled=use_cudnn,
                            _benchmark=use_cudnn_benchmark and not is_deterministic,
                            _deterministic=is_deterministic)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('use_cudnn', default='True', type=str, help='try using cudnn', is_bool=True),
            Argument('use_cudnn_benchmark', default='True', type=str, help='use cudnn benchmark', is_bool=True),
        ]
