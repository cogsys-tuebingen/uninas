from copy import deepcopy
import torch
import torch.nn as nn
from uninas.methods.abstract import AbstractMethod
from uninas.training.devices.abstract import AbstractDevicesManager
from uninas.utils.loggers.python import Logger


class ModelEMA:
    """
    Updates an Exponential Moving Average weight copy of the give 'model', using given 'decay' on the given 'device'

    Since this model uses a backward hook (before the update step), the very last update will not be applied.
    However, it also makes using this model very easy, since it is updated automatically.
    """

    devices = ['disabled', 'cpu', 'same']

    def __init__(self, model: AbstractMethod, decay=0.9999, device='same'):
        assert 1 > decay > 0
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.decay_m = 1 - decay
        self.is_same_device = device == 'same'
        self._handle = None

        # set device
        if device == 'cpu':
            self.module.cpu()
        elif device == 'same':
            device = AbstractDevicesManager.get_device(model)
            self.module.to(device)
        else:
            raise NotImplementedError('Device "%s" can not be handled' % device)
        self.device = device

        # register a hook, to trigger 'on_backward' whenever gradients are calculated
        self._handle = model.register_backward_hook(self.on_backward)

    @classmethod
    def maybe_init(cls, logger: Logger, model: AbstractMethod, decay=0.9999, device='disabled'):
        """
        :param logger:
        :param model: model which weights' to track
        :param decay: EMA decay
        :param device: device to place upon
        :return: ModelEMA if a device is given and the decay is in [0, 1]
        """
        assert device in cls.devices
        if device == cls.devices[0]:
            logger.info('Will not use an EMA model (disabled)')
        elif (decay <= 0.0) or (decay >= 1.0):
            logger.info('Will not use an EMA model (bad decay: %f)' % decay)
        else:
            logger.info('Adding an EMA model on device: %s' % device)
            return cls(model, decay, device)
        return None

    @property
    def current_epoch(self):
        return self.module.current_epoch

    def update(self, method: AbstractMethod):
        self.module.trained_epochs = method.trained_epochs
        self.module._current_epoch = method.current_epoch

    def __call__(self, *args, **kwargs):
        if self.module.testing:
            return self.module.test_step(*args, **kwargs)
        return self.module.validation_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.module.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.module.test_step(*args, **kwargs)

    def train(self):
        pass

    def eval(self):
        pass

    def on_backward(self, model: nn.Module, *_):
        with torch.no_grad():
            # parameters
            for i, ((n0, p0), (n1, p1)) in enumerate(zip(model.named_parameters(), self.module.named_parameters())):
                assert n0 == n1
                p1.mul_(self.decay).add_(p0.data.to(self.device), alpha=self.decay_m)
            # buffers, integer values have to be copied fully (e.g. BN num batches tracked)
            for i, ((n0, p0), (n1, p1)) in enumerate(zip(model.named_buffers(), self.module.named_buffers())):
                assert n0 == n1
                if p0.dtype in (torch.float64, torch.float32, torch.float16):
                    p1.mul_(self.decay).add_(p0.data.to(self.device), alpha=self.decay_m)
                else:
                    p1.data = p0.data.to(self.device)

    def stop(self):
        self._handle.remove()
        self._handle = None
