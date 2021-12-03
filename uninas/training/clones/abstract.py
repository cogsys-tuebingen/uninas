from copy import deepcopy
from uninas.methods.abstract_method import AbstractMethod, MethodWrapper
from uninas.training.devices.abstract import AbstractDevicesManager
from uninas.utils.args import ArgsInterface, Argument, Namespace


class AbstractMethodClone(MethodWrapper, ArgsInterface):
    """
    base class for clones that are updated relative to the original model (e.g. EMA weights)
    as they are not trained, they are always in eval mode
    """

    def __init__(self, device_str="cpu", is_same_device=False, **kwargs):
        super().__init__(method=None)
        self._device_str = device_str
        self._device = None
        self._is_same_device = is_same_device
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def init(self, original: AbstractMethod) -> 'AbstractMethodClone':
        """ initialize the clone from the original """

        # make a clone, move it to the correct device
        self.method = deepcopy(original)
        if self._device_str == 'same':
            self._device = AbstractDevicesManager.get_device(original)
        elif self._device_str == 'cpu':
            self._device = self._device_str
        else:
            raise NotImplementedError("device choice %s not implemented" % self._device_str)
        self.method.to(self._device)
        self.set_mode(valid=True)

        return self

    @classmethod
    def from_args(cls, args: Namespace, index: int = None) -> 'AbstractMethodClone':
        # parsed arguments, and the global save dir
        all_args = cls._all_parsed_arguments(args, index=index)
        device = all_args.pop('device')
        return cls(device_str=device, is_same_device=(device == "same"), **all_args)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('device', default='same', type=str, choices=['same', 'cpu'],
                     help='on which device to place to clone, it can only be evaluated on the same device'),
        ]

    def __call__(self, batch, batch_idx):
        return self.forward(batch, batch_idx)

    def is_on_same_device(self) -> bool:
        return self._is_same_device

    def get_name(self) -> str:
        """ name used for log dicts """
        raise NotImplementedError

    def on_update(self, original: AbstractMethod):
        """ whenever the weights of the original method are updated """
        pass

    def on_training_epoch_end(self, method: AbstractMethod):
        """ whenever the original method completed an epoch """
        self.method.training_epoch_end([])
        self.method.trained_epochs = method.trained_epochs
        self.method._current_epoch = method.current_epoch

    def stop(self):
        """ when the training stops """
        pass
