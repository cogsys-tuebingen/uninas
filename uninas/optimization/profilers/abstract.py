import os
from uninas.models.networks.uninas.search import SearchUninasNetwork
from uninas.optimization.profilers.functions import AbstractProfileFunction
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.utils.args import ArgsInterface, MetaArgument, Namespace
from uninas.utils.loggers.python import LoggerManager
from uninas.utils.paths import replace_standard_paths
from uninas.register import Register


class AbstractProfiler(ArgsInterface):
    def __init__(self, profile_fun: AbstractProfileFunction, is_test_run=False, **__):
        super().__init__()
        self.profile_fun = profile_fun
        self.is_test_run = is_test_run
        self.logger = LoggerManager().get_logger()

    def _str_dict(self) -> dict:
        dct = super()._str_dict()
        dct['profile_fun'] = self.profile_fun.__class__.__name__
        return dct

    @classmethod
    def from_args(cls, args: Namespace, index=None, is_test_run=False) -> 'AbstractProfiler':
        """ create a profiler from a argparse arguments """
        cls_profile_fun = cls._parsed_meta_argument(Register.profile_functions, 'cls_profile_fun', args, index=index)
        profile_fun = cls_profile_fun.from_args(args, index=None)
        return cls(profile_fun, is_test_run=is_test_run, **cls._all_parsed_arguments(args, index=index))

    @classmethod
    def meta_args_to_add(cls, num_optimizers=1) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + [
            MetaArgument('cls_profile_fun', Register.profile_functions, help_name='profile function', allowed_num=1),
        ]

    def save(self, dir_: str):
        """ save the profiling data in this dir """
        path = replace_standard_paths(dir_)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._save(dir_)

    def load(self, dir_: str) -> bool:
        """ load the profiling data from this dir """
        dir_ = replace_standard_paths(dir_)
        if os.path.isdir(dir_):
            self._load(dir_)
            return True
        return False

    def _save(self, dir_: str):
        """ save the profiling data in this dir """
        raise NotImplementedError

    def _load(self, dir_: str):
        """ load the profiling data from this dir """
        raise NotImplementedError

    def profile(self, network: SearchUninasNetwork, mover: AbstractDeviceMover, batch_size: int):
        """ profile the network """
        raise NotImplementedError
