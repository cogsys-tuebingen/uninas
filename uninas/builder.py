"""
This class goes through every single file in its directory and all sub-directories, storing all class references.
This enables:
- every class can register itself
- can use the names to use the registered classes/functions without knowing where they are in the project code
- can save all network configurations as text (i.e. json) and properly restore them
"""

import inspect
import json
import os
from glob import glob
import importlib
from uninas.utils.paths import replace_standard_paths
from uninas.utils.loggers.python import get_logger
from uninas.utils.meta import Singleton
from uninas.register import Register

logger = get_logger()


class Builder(metaclass=Singleton):

    def __init__(self):
        """
        A class to save+rebuild network_configs,
        automatically crawls all classes in the directory its in, including sub directories
        """
        self.classes = {}
        own_path = os.path.abspath(__file__)
        own_dir = os.path.dirname(own_path)
        offset = len(os.path.abspath(own_dir + '/../') + '/')
        ignore_dirs = ['/gui/']

        for file in glob(own_dir + '/**', recursive=True):
            skip = False
            for ignore_dir in ignore_dirs:
                if ignore_dir in file:
                    skip = True
            if (not skip) and (file != own_path) and ('__' not in file) and file.endswith('.py'):
                module_name = file[offset:-3].replace('/', '.')
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    self.classes[name] = obj
        Register.builder = self

    def add_modules(self, modules: list):
        """ add class references from external modules """
        for module in modules:
            for name, obj in inspect.getmembers(module, inspect.isclass):
                self.classes[name] = obj

    def from_config(self, config: dict):
        """ try to load a nested substructure / net from the given instructions """
        cfg = config.copy()
        name = cfg.pop('name')
        return self.classes[name].from_config(**cfg)

    def print_available_cls(self):
        for k, v in self.classes.items():
            print(k, v)

    @staticmethod
    def save_config(config: dict, config_file_path: str):
        """ save the config of the net as a json file """
        if config is not None:
            path_dir = '/'.join(config_file_path.split('/')[:-1])
            os.makedirs(path_dir, exist_ok=True)
            with open(config_file_path, 'w+') as outfile:
                json.dump(config, outfile, ensure_ascii=False, indent=2)
            logger.info('Wrote net config to %s' % config_file_path)

    @staticmethod
    def load_config(config_file_path: str) -> dict:
        """ load a json config file from given path """
        config_file_path = replace_standard_paths(config_file_path)
        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r') as file:
                return json.load(file)
        else:
            raise FileNotFoundError('Could not find file "%s" / "%s"' %
                                    (config_file_path, os.path.abspath(config_file_path)))

    def load_from_config(self, config_file_path: str):
        """ load a json config file from given path and try to restore a corresponding net topology """
        return self.from_config(self.load_config(config_file_path))

    @staticmethod
    def _rec_list_attr(dct: dict, attr_name: str) -> [str]:
        attrs = []
        if isinstance(dct, dict):
            attr = dct.get(attr_name, None)
            if attr is not None:
                attrs.append(attr)
            for k, v in dct.items():
                attrs += Builder._rec_list_attr(v, attr_name)
        elif isinstance(dct, (list, tuple)):
            for v in dct:
                attrs += Builder._rec_list_attr(v, attr_name)
        return attrs

    def find_classes_in_config(self, config_file_path: str) -> {str: list}:
        """ load a json config file from given path and figure out the used classes in it """
        config_file_path = replace_standard_paths(config_file_path)
        cfg = self.load_config(config_file_path)
        names = self._rec_list_attr(cfg, 'name')
        return {
            'cls_network_body': [n for n in names if n.endswith('NetworkBody')],
            'cls_network_cells': [n for n in names if n.endswith('Cell')],
            'cls_network_stem': [n for n in names if n.endswith('Stem')],
            'cls_network_heads': [n for n in names if n.endswith('Head')],
        }
