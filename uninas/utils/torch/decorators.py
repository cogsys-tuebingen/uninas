import torch.nn as nn
from functools import wraps


def use_eval(method):
    """ temporarily use eval mode, return to the previously used training mode afterwards again """

    @wraps(method)
    def wrapped(self: nn.Module, *args, **kwargs):
        is_training = self.training
        if is_training:
            self.eval()
        result = method(self, *args, **kwargs)
        if is_training:
            self.train()
        return result
    return wrapped
