from typing import Union
import torch
from pytorch_lightning.core.step_result import Result

log_prefix = '@log_'


class ResultValue:
    def __init__(self, value, count: int = 1):
        if isinstance(value, torch.Tensor):
            self.value = value.clone().detach()
        else:
            self.value = torch.Tensor([value])
        if len(self.value.shape) < 1:
            self.value = torch.unsqueeze(self.value, 0)
        self.count = count

    def get_scaled_value(self) -> torch.Tensor:
        return self.value * self.count

    def detach(self) -> 'ResultValue':
        self.value = self.value.detach()
        return self

    def unsqueeze(self, dim=0) -> 'ResultValue':
        self.value = self.value.unsqueeze(dim)
        return self

    def item(self):
        return self.value.item()

    def cpu(self) -> torch.Tensor:
        return self.value.cpu()

    def __repr__(self):
        return str(self.value)

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value


def add_to_result(result: Result, prefix: str, dct: dict = None):
    if isinstance(dct, dict):
        for k, v in dct.items():
            k2 = '%s%s' % (prefix, k)
            if isinstance(v, (torch.Tensor, ResultValue)):  # only accept ResultValue in the future?
                result[k2] = v
            else:
                result[k2] = torch.Tensor([v])


def get_from_result(result: Result, prefix: str) -> dict:
    dct = {}
    for k, v in result.items():
        if k.startswith(prefix):
            dct[k.replace(prefix, '')] = v
    return dct


class LogResult(Result):
    def __init__(self, loss: Union[torch.Tensor, None], log_info: dict = None):
        super().__init__(loss if (isinstance(loss, torch.Tensor) and (loss.grad_fn is not None)) else None)
        add_to_result(self, log_prefix, log_info)

    def get_loss(self) -> torch.Tensor:
        # TODO may not exist, thanks continuous lightning changes
        return self.minimize

    def _detach(self):
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, ResultValue)):
                self.__setitem__(k, v.detach())

    def get_log_info(self) -> dict:
        return get_from_result(self, log_prefix)

    @classmethod
    def split_log_dict(cls, log_dict: dict) -> ({str: torch.Tensor}, {str: int}):
        """ split a log into values and counts """
        values, counts = {}, {}
        for k, v in log_dict.items():
            if isinstance(v, ResultValue):
                values[k] = v.value
                counts[k] = v.count
            else:
                values[k] = v
                counts[k] = 1
        return values, counts

    def backward(self) -> 'LogResult':
        if isinstance(self.minimize, torch.Tensor):
            self.minimize.backward()
        return self

    def detach(self) -> 'LogResult':
        self._detach()
        return self

    def get_detached_copy(self) -> 'LogResult':
        return LogResult(self.get_loss().detach(), log_info=self.get_log_info())
