import os
import json
import inspect
from typing import Union
from glob import glob
from collections.abc import Callable
from urllib.request import urlretrieve
from enum import Enum
from uninas.utils.loggers.python import get_logger


logger = get_logger()


project_dir = '%s/' % ('/'.join(__file__.split('/')[:-3]))

global_config_path = '{}global_config.json'.format(project_dir)
if not os.path.isfile(global_config_path):
    print("-"*120)
    print("missing global config file")
    print("make a copy of global_config.example.json, name it global_config.json, and change the content if you like")
    print("these values are only the defaults that can be changed for each experiment individually")
    print("-"*120)
    raise FileNotFoundError(global_config_path)

standard_paths = {
    'path_project_dir': project_dir,

    'path_conf_net': '{}experiments/configs/networks/'.format(project_dir),
    'path_conf_net_originals': '{}experiments/configs/networks/originals/'.format(project_dir),
    'path_conf_net_discovered': '{}experiments/configs/networks/discovered/'.format(project_dir),
    'path_conf_net_search': '{}experiments/configs/networks/search/'.format(project_dir),

    'path_conf_tasks': '{}experiments/configs/tasks/'.format(project_dir),
    'path_conf_bench_tasks': '{}experiments/configs/tasks_bench/'.format(project_dir),

    # global_config
    'path_data': None,
    'path_pretrained': None,
    'path_tmp': None,
    'path_downloads_misc': None,
}
with open(global_config_path) as global_config_file:
    standard_paths.update(json.load(global_config_file))
for k, v in standard_paths.items():
    assert v is not None, "config '%s' is None, is it missing in %s?" % (k, global_config_path)


def get_class_path(cls: type) -> str:
    """ find in which file a class is implemented, return relative path """
    path = inspect.getfile(cls)
    return path[len(project_dir):]


def replace_standard_paths(path: str) -> str:
    """ replace common path wildcards in the string to avoid awkward relative paths in different files... """
    endswith = path.endswith('/')
    path = os.path.abspath(path.format(**standard_paths))
    if os.path.isdir(path) or endswith:
        path += '/'
    return path


class FileType(Enum):
    """ where to download to """
    MISC = standard_paths.get("path_downloads_misc")
    WEIGHTS = standard_paths.get("path_pretrained")
    DATA = standard_paths.get("path_data")


def maybe_download(path_or_url: str, file_type: FileType = FileType.MISC) -> Union[str, None]:
    """
    if the file does not locally exist at the given path,
    try to use a cached download, otherwise download it,
    then return the path
    """
    path_or_url = path_or_url.format(**standard_paths)
    if os.path.isfile(path_or_url):
        return path_or_url
    else:
        try:
            os.makedirs(file_type.value, exist_ok=True)
            file_path = '%s/%s' % (file_type.value, path_or_url.split('/')[-1])
            if not os.path.isfile(file_path):
                urlretrieve(path_or_url, file_path)
            return file_path
        except:
            return None


def split(string: str, cast_fun: Callable = None) -> list:
    """ split a comma-separated string into a list, maybe cast list elements to desired type """
    if string is None or len(string) == 0:
        return []
    re = list(string.replace(' ', '').split(','))
    if cast_fun is not None:
        re = [cast_fun(s) for s in re]
    return re


def get_net_config_dir(config_source: str) -> str:
    """ determine the path to put a network config """
    return {
        'original': replace_standard_paths('{path_conf_net_originals}/'),
        'discovered': replace_standard_paths('{path_conf_net_discovered}/'),
    }.get(config_source, '__unknown_source__')


def find_all_files(dir_: str, extension='.t7') -> list:
    """ finds all files with given extension """
    file_paths = []
    for sub_dir, _, file_names in os.walk(dir_):
        for file_name in file_names:
            if file_name.endswith(extension):
                file_paths.append('%s/%s' % (sub_dir, file_name))
    return file_paths


def find_net_config_path(config_name: str) -> str:
    """ find a network config, get its full path """
    if '/' in config_name:
        return config_name
    p = replace_standard_paths('{path_conf_net}') + '/*/' + config_name + '*'
    paths = glob(p)
    if len(paths) != 1:
        raise FileNotFoundError('must have exactly one config in %s, found %s' % (p, str(paths)))
    return paths[0]


def find_net_config_name(config_name: str) -> str:
    """ find a network config, return its name """
    cfg_path = find_net_config_path(config_name)
    return cfg_path.split('/')[-1].split('.')[0]


def find_pretrained_weights_path(path: str, name: str, raise_missing=True) -> str:
    """
    find path to pretrained weights in folder or URL
    """
    maybe_path = maybe_download(path, FileType.WEIGHTS)
    if isinstance(maybe_path, str):
        return maybe_path
    path = replace_standard_paths(path)
    if len(path) == 0 or os.path.isfile(path):
        return path
    p = '%s/**/*%s*' % (path, name)
    paths = glob(p, recursive=True)
    if len(paths) < 1:
        if raise_missing:
            raise FileNotFoundError('can not find pretrained weights in %s' % p)
        return ''
    return paths[0]
