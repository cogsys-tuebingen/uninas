from pytorch_lightning.callbacks.base import Callback
from uninas.methods.abstract_method import AbstractMethod
from uninas.training.trainer.abstract import AbstractTrainerFunctions
from uninas.utils.paths import replace_standard_paths
from uninas.utils.args import ArgsInterface, Namespace


class AbstractCallback(ArgsInterface, Callback):

    def __init__(self, save_dir: str, index: int, **_):
        super().__init__()
        self._save_dir = replace_standard_paths(save_dir)
        assert isinstance(index, int)
        self._index = index

    def setup(self, trainer: AbstractTrainerFunctions, pl_module: AbstractMethod, stage: str):
        """ Called when fit or test begins """
        pass

    def teardown(self, trainer: AbstractTrainerFunctions, pl_module: AbstractMethod, stage: str):
        """ Called when fit or test ends """
        pass

    def on_train_epoch_start(self, trainer: AbstractTrainerFunctions,
                             pl_module: AbstractMethod,
                             log_dict: dict = None):
        """ Called when the train epoch begins. """
        pass

    def on_train_epoch_end(self, trainer: AbstractTrainerFunctions,
                           pl_module: AbstractMethod,
                           log_dict: dict = None):
        """ Called when the train epoch ends. """
        pass

    def on_validation_epoch_end(self, trainer: AbstractTrainerFunctions,
                                pl_module: AbstractMethod,
                                log_dict: dict = None):
        """ Called when the val epoch ends. """
        pass

    def on_test_epoch_end(self, trainer: AbstractTrainerFunctions,
                          pl_module: AbstractMethod,
                          log_dict: dict = None):
        """ Called when the test epoch ends. """
        pass

    @classmethod
    def from_args(cls, save_dir: str, args: Namespace, index: int) -> 'AbstractCallback':
        parsed = cls._all_parsed_arguments(args, index=index)
        return cls(save_dir, index, **parsed)

    def _dict_key(self, key: str) -> str:
        return 'callback/%d/%s/%s' % (self._index, self.__class__.__name__, key)


# this is only here to avoid registering the checkpoint callback class multiple times
class EpochInfo:
    def __init__(self):
        self.epoch = -1
        self.log_dict = {}
        self.checkpoint_path = None

    def __repr__(self):
        return "%s(epoch=%d, path=%s, log_dict=%s)" %\
               (self.__class__.__name__, self.epoch, self.checkpoint_path, self.log_dict)
