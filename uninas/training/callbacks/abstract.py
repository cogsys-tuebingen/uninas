from pytorch_lightning.callbacks.base import Callback
from uninas.training.trainer.abstract import AbstractTrainerFunctions
from uninas.methods.abstract import AbstractMethod
from uninas.utils.torch.ema import ModelEMA
from uninas.utils.paths import replace_standard_paths
from uninas.utils.args import ArgsInterface, Namespace


class AbstractCallback(ArgsInterface, Callback):

    def __init__(self, save_dir: str, index: int, **_):
        super().__init__()
        self._save_dir = replace_standard_paths(save_dir)
        self._index = index

    def on_set_method(self, trainer: AbstractTrainerFunctions, pl_module: AbstractMethod):
        """ called when the trainer changes the method it trains (also called for the first one) """
        pass

    def setup(self, trainer: AbstractTrainerFunctions, pl_module: AbstractMethod, stage: str):
        """Called when fit or test begins"""
        pass

    def teardown(self, trainer: AbstractTrainerFunctions, pl_module: AbstractMethod, stage: str):
        """Called when fit or test ends"""
        pass

    def on_train_epoch_start(self, trainer: AbstractTrainerFunctions,
                             pl_module: AbstractMethod,
                             pl_ema_module: ModelEMA = None,
                             log_dict: dict = None):
        """Called when the train epoch begins."""
        pass

    def on_train_epoch_end(self, trainer: AbstractTrainerFunctions,
                           pl_module: AbstractMethod,
                           pl_ema_module: ModelEMA = None,
                           log_dict: dict = None):
        """Called when the train epoch ends."""
        pass

    def on_validation_epoch_end(self, trainer: AbstractTrainerFunctions,
                                pl_module: AbstractMethod,
                                pl_ema_module: ModelEMA = None,
                                log_dict: dict = None):
        """Called when the val epoch ends."""
        pass

    def on_test_epoch_end(self, trainer: AbstractTrainerFunctions,
                          pl_module: AbstractMethod,
                          pl_ema_module: ModelEMA = None,
                          log_dict: dict = None):
        """Called when the test epoch ends."""
        pass

    @classmethod
    def get_method(cls, method: AbstractMethod, method_ema: ModelEMA = None, prefer_ema=True) -> AbstractMethod:
        if isinstance(method_ema, ModelEMA) and prefer_ema:
            return method_ema.module
        return method

    @classmethod
    def from_args(cls, save_dir: str, args: Namespace, index: int):
        parsed = cls._all_parsed_arguments(args, index=index)
        return cls(save_dir, index, **parsed)


# this is only here to avoid registering the checkpoint callback class multiple times
class EpochInfo:
    def __init__(self):
        self.epoch = -1
        self.log_dict = {}
        self.checkpoint_path = None
