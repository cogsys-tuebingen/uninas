import os
import torch
from uninas.networks.self.search import SearchUninasNetwork
from uninas.optimization.common.profilers.functions import AbstractProfileFunction
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.utils.args import ArgsInterface, MetaArgument, Namespace
from uninas.utils.loggers.python import get_logger
from uninas.register import Register


class AbstractProfiler(ArgsInterface):
    def __init__(self, profile_fun: AbstractProfileFunction = None, is_test_run=False, **__):
        super().__init__()
        self.data = dict(meta=dict(cls=self.__class__.__name__))
        if profile_fun is not None:
            self.set_all(profile_cls=profile_fun.__class__.__name__, is_test_run=is_test_run)
        self.profile_fun = profile_fun
        self.logger = get_logger()

    @property
    def name(self):
        return '%s.%s' % (self.__class__.__name__, str(self.get('profile_cls')))

    @classmethod
    def from_args(cls, args: Namespace, index=None, is_test_run=False):
        """ create a profiler from a argparse arguments """
        profile_fun = None
        try:
            cls_profile_fun = cls._parsed_meta_argument('cls_profile_fun', args, index=index)
            profile_fun = cls_profile_fun.from_args(args, index=None)
        except KeyError:
            pass
        return cls(profile_fun=profile_fun, is_test_run=is_test_run, **cls._all_parsed_arguments(args, index=index))

    @classmethod
    def from_file(cls, file_path: str):
        """ create and load a profiler from a profiler save file """
        assert os.path.isfile(file_path), "File does not exist: %s" % str(file_path)
        cls_name = torch.load(file_path).get('meta').get('cls')
        profiler = Register.get(cls_name)()
        profiler.load(file_path)
        if profiler.get('is_test_run'):
            get_logger().warning("Loading profiler data from a test-run file!")
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
        torch.save(self.data, path)

    def load(self, path: str) -> bool:
        if os.path.isfile(path):
            self.data.update(torch.load(path))
            return True
        return False

    def profile(self, network: SearchUninasNetwork, mover: AbstractDeviceMover, batch_size: int):
        """ profile the network """
        raise NotImplementedError

    def predict(self, values: tuple) -> float:
        """ predict the network's profile value with the given architecture """
        raise NotImplementedError
