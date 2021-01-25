import os
import torch
from uninas.networks.uninas.search import SearchUninasNetwork
from uninas.optimization.common.profilers.functions import AbstractProfileFunction
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.utils.args import ArgsInterface, MetaArgument, Namespace
from uninas.utils.loggers.python import LoggerManager
from uninas.utils.paths import replace_standard_paths
from uninas.register import Register


class AbstractProfiler(ArgsInterface):
    def __init__(self, profile_fun: AbstractProfileFunction = None, is_test_run=False, **__):
        super().__init__()
        self.data = dict(meta=dict(cls=self.__class__.__name__))
        if profile_fun is not None:
            self.set_all(profile_cls=profile_fun.__class__.__name__, is_test_run=is_test_run)
        self.profile_fun = profile_fun
        self.logger = LoggerManager().get_logger()

    @property
    def name(self):
        return '%s.%s' % (self.__class__.__name__, str(self.get('profile_cls')))

    @classmethod
    def from_args(cls, args: Namespace, index=None, is_test_run=False) -> 'AbstractProfiler':
        """ create a profiler from a argparse arguments """
        profile_fun = None
        try:
            cls_profile_fun = cls._parsed_meta_argument(Register.profile_functions, 'cls_profile_fun', args, index=index)
            profile_fun = cls_profile_fun.from_args(args, index=None)
        except KeyError:
            pass
        return cls(profile_fun=profile_fun, is_test_run=is_test_run, **cls._all_parsed_arguments(args, index=index))

    @classmethod
    def from_file(cls, file_path: str):
        """ create and load a profiler from a profiler save file """
        file_path = replace_standard_paths(file_path)
        assert os.path.isfile(file_path), "File does not exist: %s" % str(file_path)
        cls_name = torch.load(file_path).get('meta').get('cls')
        profiler = Register.profilers.get(cls_name)()
        profiler.load(file_path)
        if profiler.get('is_test_run'):
            LoggerManager().get_logger().warning("Loading profiler data from a file created in a test run!")
        return profiler

    @classmethod
    def meta_args_to_add(cls, num_optimizers=1) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + [
            MetaArgument('cls_profile_fun', Register.profile_functions, help_name='profile function', allowed_num=1),
        ]

    def set(self, key: str, value):
        self.data['meta'][key] = value

    def set_all(self, **key_value_pairs):
        for key, value in key_value_pairs.items():
            self.set(key, value)

    def get(self, key: str):
        return self.data['meta'].get(key)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._get_data_to_save(), path)

    def _get_data_to_save(self) -> dict:
        return self.data

    def load(self, path: str) -> bool:
        if os.path.isfile(path):
            data = torch.load(path)
            assert isinstance(data, dict)
            self._load_data(data)
            return True
        return False

    def _load_data(self, data: dict):
        self.data.update(data)

    def profile(self, network: SearchUninasNetwork, mover: AbstractDeviceMover, batch_size: int):
        """ profile the network """
        raise NotImplementedError

    def predict(self, values: tuple) -> float:
        """ predict the network's profile value with the given architecture """
        raise NotImplementedError
