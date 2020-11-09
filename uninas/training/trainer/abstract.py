from typing import Union, Iterable, Tuple
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.nn.parallel import DistributedDataParallel as Ddp
from uninas.methods.abstract import AbstractMethod
from uninas.networks.abstract import AbstractNetwork
from uninas.training.optimizers.abstract import AbstractOptimizer
from uninas.training.regularizers.abstract import AbstractRegularizer
from uninas.utils.torch.ema import ModelEMA


class AbstractTrainerFunctions:
    """
    to avoid cyclic imports with the actual trainer implementations while still ensuring some methods
    (e.g. in callbacks)
    """

    def get_method(self) -> AbstractMethod:
        raise NotImplementedError

    def get_network(self) -> AbstractNetwork:
        return self.get_method().get_network()

    def get_regularizers(self) -> [AbstractRegularizer]:
        """ get regularizers """
        return self.get_method().get_regularizers()

    def get_optimizers(self) -> [Optimizer]:
        """ get optimizers """
        raise NotImplementedError

    def get_optimizer_log_dict(self) -> dict:
        return AbstractOptimizer.get_optimizer_log_dict(self.get_optimizers())

    @classmethod
    def choose_method(cls, method: AbstractMethod, method_ema: Union[ModelEMA, None], prefer_ema=True) -> AbstractMethod:
        """ get module or module_ema, using preference and avoiding None """
        ms = [method_ema, method] if prefer_ema else [method, method_ema]
        for m in ms:
            if m is None:
                continue
            if isinstance(m, (Ddp, ModelEMA)):
                return m.module
            return m

    @classmethod
    def iterate_usable_methods(cls, method: nn.Module, method_ema: Union[ModelEMA, None]) -> Iterable[Tuple[AbstractMethod, str]]:
        """
        iterate the methods that can be used for forward passes

        :param method:
        :param method_ema:
        :return: pairs of (method, format string for log_dicts)
        """
        if isinstance(method, nn.Module):
            yield method, '%s'
        if isinstance(method_ema, ModelEMA) and method_ema.is_same_device:
            yield method_ema, '%s_ema'

    def get_checkpoint_update_dict(self, *_) -> dict:
        """ get the internal state """
        # optional argument required for lightning
        return {'trainer_state': self._get_state_dict()}

    def _get_state_dict(self) -> dict:
        """ get the internal state """
        return dict()

    def _load_state_dict(self, state: dict):
        """ load the internal state """
        pass
