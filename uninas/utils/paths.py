import os
import json
import inspect
from typing import Union
from urllib.request import urlretrieve
from enum import Enum, auto
from uninas.utils.loggers.python import LoggerManager


name_task_config = 'task.run_config'
standard_paths = {}


def initialize_paths(global_config_dir: str = None):
    global standard_paths
    code_dir = '%s/' % '/'.join(__file__.split('/')[:-3])

    # default paths for uninas files/dirs
    standard_paths.update({
        # location where to find the current global config file
        'path_config_dir': global_config_dir if isinstance(global_config_dir, str) else code_dir,

        # location of the uninas code
        'path_code_dir': code_dir,

        'path_conf_net': '{}experiments/configs/networks/'.format(code_dir),
        'path_conf_net_originals': '{}experiments/configs/networks/originals/'.format(code_dir),
        'path_conf_net_discovered': '{}experiments/configs/networks/discovered/'.format(code_dir),
        'path_conf_net_search': '{}experiments/configs/networks/search/'.format(code_dir),

        'path_conf_tasks': '{}experiments/configs/tasks/'.format(code_dir),
        'path_conf_bench_tasks': '{}experiments/configs/tasks_bench/'.format(code_dir),

        # set in the global_config
        'path_data': None,
        'path_pretrained': None,
        "path_profiled": None,
        'path_tmp': None,
        'path_downloads_misc': None,
    })

    # adding paths from the global config
    if global_config_dir is None:
        global_config_dir = code_dir
    if not global_config_dir.endswith('/'):
        global_config_dir += '/'

    # make sure all paths are set, otherwise instruct to do so
    global_config_path = '{}global_config.json'.format(global_config_dir)
    if not os.path.isfile(global_config_path):
        print("-"*120)
        print("missing global config file")
        print("make a copy of global_config.example.json, name it global_config.json, and change the content if you like")
        print("these values are only the defaults that can be changed for each experiment individually")
        print("-"*120)
        raise FileNotFoundError(global_config_path)
    print('loading global config from %s' % global_config_path)

    with open(global_config_path) as global_config_file:
        standard_paths.update(json.load(global_config_file))
    for k, v in standard_paths.items():
        assert v is not None, "config '%s' is None, is it missing in %s?" % (k, global_config_path)


def get_class_path(cls: type) -> str:
    """ find in which file a class is implemented, return relative path """
    path = inspect.getfile(cls)
    # return path[len(project_dir):]
    return path


def replace_standard_paths(path: str, make_dir=False) -> str:
    """ replace common path wildcards in the string to avoid awkward relative paths in different files... """
    endswith = path.endswith('/')
    path = os.path.expanduser(path)
    if path.startswith("."):
        path = os.path.abspath(path)
    path = path.format(**standard_paths)
    if make_dir:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isdir(path) or endswith:
        path += '/'
    return path


def make_base_dirs(path: str) -> str:
    """ create the dir and return the path """
    path = replace_standard_paths(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


class FileType(Enum):
    """ where to download to """
    MISC = "path_downloads_misc"
    WEIGHTS = "path_pretrained"
    DATA = "path_data"

    @property
    def path(self) -> str:
        return standard_paths.get(self.value)


def maybe_download(path_or_url: str, file_type: FileType = FileType.MISC) -> Union[str, None]:
    """
    if the file does not locally exist at the given path,
    try to use a cached download, otherwise download it,
    then return the path
    """
    path_or_url = replace_standard_paths(path_or_url)
    if os.path.isfile(path_or_url):
        return path_or_url
    else:
        try:
            os.makedirs(file_type.path, exist_ok=True)
            file_path = '%s/%s' % (file_type.path, path_or_url.split('/')[-1])
            if not os.path.isfile(file_path):
                urlretrieve(path_or_url, file_path)
                LoggerManager().get_logger().info("downloaded %s to %s" % (path_or_url, file_path))
            return file_path
        except:
            return None


def get_task_config_path(dir_: str) -> str:
    return "%s/%s" % (replace_standard_paths(dir_), name_task_config)


def get_net_config_dir(config_source: str) -> str:
    """ determine the path to put a network config """
    splits = config_source.split('/')
    splits[0] = {
        'original': replace_standard_paths('{path_conf_net_originals}/'),
        'originals': replace_standard_paths('{path_conf_net_originals}/'),
        'discovered': replace_standard_paths('{path_conf_net_discovered}/'),
    }.get(splits[0], '__unknown_source__')
    return '/'.join(splits)


def find_all_files(dir_: str, extension='.pt') -> list:
    """ finds all files with given extension """
    file_paths = []
    for sub_dir, _, file_names in os.walk(replace_standard_paths(dir_)):
        for file_name in file_names:
            if file_name.endswith(extension):
                file_paths.append('%s/%s' % (sub_dir, file_name))
    return file_paths
