from collections import Callable
from uninas.methods.abstract import AbstractMethod
from uninas.training.trainer.abstract import AbstractTrainerFunctions
from uninas.training.optimizers.abstract import AbstractOptimizer
from uninas.training.callbacks.abstract import AbstractCallback
from uninas.training.callbacks.checkpoint import CheckpointCallback
from uninas.utils.torch.ema import ModelEMA


class PbtServerResponse:
    """
    A response sent by the server
    """

    def __init__(self, client_id: int = 0, save_ema=False):
        self.client_id = client_id
        self.save_ema = save_ema
        self.save_path = None
        self.load_path = None
        self.optimizer_lrs = {}         # {optimizer index: new lr}
        self.reqularizer_values = {}    # {regularizer name: new value}

    def act(self, callback: AbstractCallback, log_fun: Callable, trainer: AbstractTrainerFunctions,
            pl_module: AbstractMethod, pl_ema_module: ModelEMA = None):
        """
        lazy way of handling the response, so that the related code is in one place
        """
        # saving model weights
        if isinstance(self.save_path, str):
            log_fun("saving to: %s, prefer ema=%s" % (self.save_path, str(self.save_ema)))
            CheckpointCallback.save(self.save_path,
                                    callback.get_method(pl_module, pl_ema_module, prefer_ema=self.save_ema))
        # loading model weights
        if isinstance(self.load_path, str):
            log_fun("loading from: %s" % self.load_path)
            CheckpointCallback.wait_load(self.load_path)

        # learning rates
        if len(self.optimizer_lrs) > 0:
            optimizers = trainer.get_optimizers()
            for optimizer_id, v in self.optimizer_lrs.items():
                log_fun("setting learning rate of optimizer %d to %f" % (optimizer_id, v))
                AbstractOptimizer.set_optimizer_lr_by_index(optimizers, optimizer_id, lr=v, is_multiplier=False)

        # regularizer values
        for k, v in self.reqularizer_values.items():
            for regularizer in trainer.get_regularizers():
                if regularizer.__class__.__name__ == k:
                    log_fun("setting %s value to %s" % (k, str(v)))
                    regularizer.set_value(v)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_dict(cls, dct: dict):
        r = PbtServerResponse()
        r.__dict__.update(dct)
        return r
