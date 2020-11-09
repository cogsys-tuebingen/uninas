"""
regularizing networks during training
"""

from uninas.model.networks.abstract import AbstractNetworkBody
from uninas.utils.args import ArgsInterface, Namespace


class AbstractRegularizer(ArgsInterface):
    def __init__(self, _: Namespace, index=None):
        ArgsInterface.__init__(self)
        self.index = index
        self._changed = True

    def on_start(self, max_epochs: int, net: AbstractNetworkBody) -> dict:
        return {}

    def on_epoch_start(self, cur_epoch: int, max_epochs: int, net: AbstractNetworkBody) -> dict:
        return {}

    def on_epoch_end(self, cur_epoch: int, max_epochs: int, net: AbstractNetworkBody) -> dict:
        return {}

    def set_value(self, v):
        """ externally set the main value of this regularizer """
        self._set_value(v)
        self._changed = True

    def _set_value(self, v):
        """ externally set the main value of this regularizer """
        raise NotImplementedError

    @classmethod
    def _dict_key(cls, key: str) -> str:
        return 'regularization/%s/%s' % (cls.__name__, key)

    @classmethod
    def filter_values_in_dict(cls, log_dict: dict, regularizer_name: str) -> dict:
        """ return only the log_dict entries that match this regularizer """
        filtered, s = {}, ("regularization/%s" % regularizer_name)
        for k, v in log_dict.items():
            if k.startswith(s):
                filtered[k] = v
        return filtered
