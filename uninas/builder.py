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
from uninas.utils.paths import initialize_paths, replace_standard_paths
from uninas.utils.loggers.python import LoggerManager
from uninas.utils.meta import Singleton
from uninas.register import Register


class Builder(metaclass=Singleton):
    extension_net_config = '.network_config'

    def __init__(self, ignore_files=(), global_config_dir: str = None):
        """
        A class to save+rebuild network_configs,
        automatically crawls all classes in the directory its in, including sub directories

        :param ignore_files:
            useful to ignore the current file, if the Builder is created for the first time
            (i.e. due to a locally run __main__ there) in a file where also something is registered,
            to avoid an error due to registering it twice
        :param global_config_dir:
            dir used to locate and load the global config for path replacements
            can be set manually to enable other projects extending the code base
        """
        initialize_paths(global_config_dir=global_config_dir)
        self.classes = {}
        Register.builder = self

        own_path = os.path.abspath(__file__)
        own_dir = os.path.dirname(own_path)
        ignore_dirs = ['/gui/']
        ignore_files = [own_path] + list(ignore_files)
        self.add_dir(own_dir, ignore_files=ignore_files, ignore_dirs=ignore_dirs)

    def add_modules(self, modules: list):
        """ add class references from external modules """
        for module in modules:
            for name, obj in inspect.getmembers(module, inspect.isclass):
                self.classes[name] = obj

    def add_dir(self, dir_: str, ignore_files=(), ignore_dirs=()):
        offset = len(os.path.abspath(dir_ + '/../') + '/')

        for file in glob(dir_ + '/**', recursive=True):
            if file in ignore_files:
                continue
            skip = False
            for ignore_dir in ignore_dirs:
                if ignore_dir in file:
                    skip = True
                    break
            if skip:
                continue
            if ('__' not in file) and file.endswith('.py'):
                module_name = file[offset:-3].replace('/', '.')
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    self.classes[name] = obj

    def from_config(self, config: dict):
        """ try to load a nested substructure / net from the given instructions """
        cfg = config.copy()
        name = cfg.pop('name')
        return self.classes[name].from_config(**cfg)

    def load_from_config(self, config_file_path: str):
        """ load a json config file from given path and try to restore a corresponding net topology """
        return self.from_config(self.load_config(config_file_path))

    def print_available_cls(self):
        for k, v in self.classes.items():
            print(k, v)

    @classmethod
    def save_config(cls, config: dict, config_dir: str, config_name: str) -> str:
        if config is None:
            return ''
        os.makedirs(config_dir, exist_ok=True)
        path = '%s/%s%s' % (replace_standard_paths(config_dir), config_name, cls.extension_net_config)
        path = os.path.abspath(path)
        with open(path, 'w+') as outfile:
            json.dump(config, outfile, ensure_ascii=False, indent=2)
        LoggerManager().get_logger().info('Wrote net config to %s' % path)
        return path

    @classmethod
    def load_config(cls, config_file_path: str) -> dict:
        """ load a json config file from given path """
        config_file_path = replace_standard_paths(config_file_path)
        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r') as file:
                return json.load(file)
        else:
            raise FileNotFoundError('Could not find file "%s" / "%s"' %
                                    (config_file_path, os.path.abspath(config_file_path)))

    @classmethod
    def find_net_config_path(cls, config_name_or_dir: str, pattern: str = '') -> str:
        """
        find a standard network config, get its full path, try:
            1) look up if config_name_or_dir matches a name (e.g. DARTS) in the common configs
            2) otherwise assume it is a dir, try to return a network config which name contains the pattern

        :param config_name_or_dir: dir or full path of a config
        :param pattern: if a config has to be searched in the dir 
        """
        # already done?
        if '/' in config_name_or_dir and config_name_or_dir.endswith(cls.extension_net_config):
            return config_name_or_dir
        # search in the common configs for a given name
        p = replace_standard_paths('{path_conf_net}') + '/**/' + config_name_or_dir + '*' + cls.extension_net_config
        paths = glob(p, recursive=True)
        if len(paths) == 1:
            return paths[0]
        # find any unique network config in the given path
        p = '%s/**/*%s*%s' % (replace_standard_paths(config_name_or_dir), pattern, cls.extension_net_config)
        paths = glob(p, recursive=True)
        if len(paths) == 1:
            return paths[0]
        raise FileNotFoundError('can not find (unique) network config with given name/path "%s"' % config_name_or_dir)

    @classmethod
    def net_config_name(cls, config_name_or_path: str) -> str:
        """ find a network config, return its name """
        cfg_path = cls.find_net_config_path(config_name_or_path)
        return cfg_path.split('/')[-1].split('.')[0]

    @classmethod
    def _rec_list_attr(cls, dct: dict, attr_name: str) -> [str]:
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
