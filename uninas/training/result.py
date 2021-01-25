import torch
from pytorch_lightning.core.step_result import Result, TrainResult, EvalResult

prefix = '@log_'


def add_to_result(result: Result, dct: dict = None):
    if isinstance(dct, dict):
        for k, v in dct.items():
            k2 = '%s%s' % (prefix, k)
            if isinstance(v, torch.Tensor):
                result[k2] = v
            else:
                result[k2] = torch.Tensor([v])


def get_from_result(result: Result) -> dict:
    dct = {}
    for k, v in result.items():
        if k.startswith(prefix):
            dct[k.replace(prefix, '')] = v
    return dct


class TrainLogResult(TrainResult):
    def __init__(self, loss: torch.Tensor = None, log_info: dict = None):
        super().__init__(minimize=loss)
        add_to_result(self, log_info)

    def get_log_info(self) -> dict:
        return get_from_result(self)


class EvalLogResult(EvalResult):
    def __init__(self, loss: torch.Tensor = None, log_info: dict = None):
        super().__init__(checkpoint_on=loss)
        add_to_result(self, log_info)
        self.detach()

    def get_log_info(self) -> dict:
        return get_from_result(self)
