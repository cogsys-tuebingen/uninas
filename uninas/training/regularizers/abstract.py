"""
regularizing networks during training
"""

from uninas.models.networks.abstract import AbstractNetwork
from uninas.utils.args import ArgsInterface, Namespace


class AbstractRegularizer(ArgsInterface):
    def __init__(self, _: Namespace, index: int):
        ArgsInterface.__init__(self)
        assert isinstance(index, int)
        self.index = index
        self._changed = True

    def on_start(self, max_epochs: int, net: AbstractNetwork) -> dict:
        return {}

    def on_epoch_start(self, cur_epoch: int, max_epochs: int, net: AbstractNetwork) -> dict:
        return {}

    def on_epoch_end(self, cur_epoch: int, max_epochs: int, net: AbstractNetwork) -> dict:
        return {}

    def set_value(self, v):
        """ externally set the main value of this regularizer """
        self._set_value(v)
        self._changed = True

    def _set_value(self, v):
        """ externally set the main value of this regularizer """
        raise NotImplementedError

    def _dict_key(self, key: str) -> str:
        return 'regularizer/%d/%s/%s' % (self.index, self.__class__.__name__, key)

    @classmethod
    def filter_values_in_dict(cls, log_dict: dict, regularizer_name: str) -> dict:
        """ return only the log_dict entries that match this regularizer """
        filtered, s = {}, ("regularization/%s" % regularizer_name)
        for k, v in log_dict.items():
            if k.startswith(s):
                filtered[k] = v
        return filtered
