from collections import Callable
from uninas.training.trainer.abstract import AbstractTrainerFunctions
from uninas.training.optimizers.abstract import WrappedOptimizer
from uninas.training.callbacks.checkpoint import CheckpointCallback


class PbtServerResponse:
    """
    A response sent by the server
    """

    def __init__(self, client_id: int = 0, save_clone=False):
        self.client_id = client_id
        self.save_clone = save_clone
        self.save_path = None
        self.load_path = None
        self.optimizer_lrs = {}         # {optimizer index: new lr}
        self.regularizer_values = {}    # {regularizer name: new value}

    def act(self, log_fun: Callable, trainer: AbstractTrainerFunctions):
        """
        lazy way of handling the response, so that the related code is in one place
        """
        # saving model weights
        if isinstance(self.save_path, str):
            log_fun("saving to: %s, prefer clone=%s" % (self.save_path, str(self.save_clone)))
            CheckpointCallback.save(self.save_path, trainer.choose_method(prefer_clone=self.save_clone))
        # loading model weights
        if isinstance(self.load_path, str):
            log_fun("loading from: %s" % self.load_path)
            CheckpointCallback.wait_load(self.load_path)

        # learning rates
        if len(self.optimizer_lrs) > 0:
            optimizers = trainer.get_optimizers()
            for optimizer_id, v in self.optimizer_lrs.items():
                log_fun("setting learning rate of optimizer %d to %f" % (optimizer_id, v))
                WrappedOptimizer.set_optimizer_lr_by_index(optimizers, optimizer_id, lr=v, is_multiplier=False)

        # regularizer values
        for k, v in self.regularizer_values.items():
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
