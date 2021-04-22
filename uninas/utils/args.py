"""
interface for classes (network, runner, dataset, method, cells, ...) to define + parse their args via argparse
"""

import os
import json
from typing import Union
from collections import defaultdict
from argparse import ArgumentParser, Namespace
from uninas.utils.paths import standard_paths, replace_standard_paths
from uninas.utils.misc import split
from uninas.register import Register, RegisterDict

args_type = Union[Namespace, dict]


def items(args: args_type) -> dict:
    if isinstance(args, dict):
        return args
    elif isinstance(args, Namespace):
        return args.__dict__
    raise NotImplementedError


def _arg_name(cls, name: str, cls_index=None, add_cls_prefix=True) -> str:
    """ generating keys/names for argparse arguments """
    _add_cls = cls is not None and add_cls_prefix
    _add_idx = cls_index is not None and _add_cls
    return '{cls}{index}{name}'.format(**{
        'cls': cls.__name__ if _add_cls else '',
        'index': '#%d' % cls_index if _add_idx else '',
        'name': ('.%s' if _add_cls else '%s') % name,
    })


def find_in_args(args: args_type, suffix: str) -> tuple:
    """ go through args, return value of key that ends with given suffix """
    for k, v in items(args).items():
        if k.endswith(suffix):
            return k, v
    raise ValueError('Value with suffix %s is not in args' % suffix)


def find_in_args_list(args_list: [str], keys: [str]) -> str:
    """ find the last value in the args list that matches any of the given keys """
    value = None
    for arg in args_list:
        k, v = arg.split('=')
        k = k.replace('-', '')
        if k in keys:
            value = v
    return value


def all_meta_args(args_list: [str] = None, args_in_file: dict = None) -> [str]:
    """ find all meta arguments in a list of arguments ["--cls_x=y"] or dict {"cls_x": y} """
    meta_args = []
    if isinstance(args_list, (list, tuple)):
        for a in args_list:
            if a.startswith('--cls_'):
                meta_args.append(a.split('=')[0].replace('--', ''))
    if isinstance(args_in_file, dict):
        for k in args_in_file.keys():
            if k.startswith('cls_'):
                meta_args.append(k)
    return meta_args


def save_as_json(args: args_type, file_path: str, wildcards: dict):
    """ save the given Namespace as ordered json file, replacing names with wildcards """
    file_path = replace_standard_paths(file_path)
    name_to_wildcard = {'%s.' % v: '{%s}.' % k for k, v in wildcards.items()}

    # generate a run_config from current args, replace with wildcards, sort in order of cls_* meta args
    config, config_sorted = {}, {}
    for k, v in items(args).items():
        for k2, v2 in name_to_wildcard.items():
            k = k.replace(k2, v2)
        config[k] = v
    for i, name in enumerate(config.keys()):
        if name.startswith('cls_'):
            for k, v in config.items():
                if k.startswith(name) or k.startswith('{%s' % name):
                    config_sorted[k] = v

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w+') as outfile:
        json.dump(config_sorted, outfile, indent=4)
    pass


def arg_list_from_json(paths: str) -> [str]:
    args = []
    for path in split(paths):
        path = replace_standard_paths(path)
        print('using config file: %s' % path)
        with open(path) as config_file:
            config = json.load(config_file)
            for k, v in config.items():
                args.append('--%s=%s' % (k, v))
    return args


def replace_wildcards_in_args_list(args_list: [str], wildcards: dict) -> ([str], [str]):
    including_paths = wildcards.copy()
    including_paths.update(standard_paths)
    # replace wildcards in args
    failed_args = []
    for i in range(len(args_list)):
        try:
            args_list[i] = args_list[i].format(**including_paths)
        except:
            failed_args.append(args_list[i])
    return args_list, failed_args


def sanitize(value):
    if isinstance(value, str):
        value = value.replace('"', '').replace("'", '')
    return value


class Argument:
    all = {}

    def __init__(self, name, default, type, help: str = '', choices: list = None, registered: list = None,
                 is_path=False, is_bool=False, is_fixed=False):
        """
        An argument for argparse, with additional post-parsing instructions

        :param name:
        :param default:
        :param type:
        :param help:
        :param choices: parsed value must be in choices, if choices is not None
        :param registered: optional list of registered classes
        :param is_path: if True, expands ~ / ${HOME} in the given string
        :param is_bool: if True, replaces the given string with True/False, depending if the string starts with t/T
        :param is_fixed: if True, the name is fixed and not influenced by class/index
        """
        self.name = name
        self.registered_name = None
        self.default = default
        self.type = type
        self.help = help
        self.choices = choices
        self.registered = registered
        self.is_path = is_path
        self.is_bool = is_bool
        self.is_fixed = is_fixed

    @property
    def kwargs(self) -> dict:
        return dict(default=self.default, type=self.type, help=self.help, choices=self.choices)

    def value(self, args: args_type):
        """ find the value in the global arg dict, apply the parsing rules """
        return self.apply_rules(items(args).get(self.registered_name))

    def apply_rules(self, value):
        """ apply the parsing rules to a given value """
        value = sanitize(value)
        if isinstance(value, str):
            if self.is_path:
                value = os.path.expanduser(value.replace('${HOME}', '~'))
            if self.is_bool:
                value = value.lower().startswith('t') or value == '1'
        return value

    def register(self, parser: ArgumentParser, registering_cls, index=None):
        """ register this argument with argparse """
        self.registered_name = self.name if self.is_fixed else _arg_name(registering_cls, self.name, index)
        self.all[self.registered_name] = self
        parser.add_argument('--%s' % self.registered_name, **self.kwargs)

    @classmethod
    def reset_cached(cls):
        cls.all = {}


class MetaArgument:
    def __init__(self, name: str, registered: RegisterDict, help_name='', allowed_num=(0, -1),
                 allow_duplicates=False, use_index=None, optional_for_loading=False):
        """
        a meta argument, an argument that specifies which classes will add further arguments

        :param name: name of the argument (e.g. "cls_trainer")
        :param registered: Register dict classes
        :param help_name: name in the help string
        :param allowed_num: int or (int, int), number of allowed values for the argument
        :param allow_duplicates: whether multiple specified classes may refer to the same python class
        :param use_index: force an index if True, even if the allowed number is limited to 1, never use an index if False
        :param optional_for_loading: whether this argument is optional when loading an existing config
        """

        self.registered = registered
        if isinstance(allowed_num, int):
            allowed_num = (allowed_num, allowed_num)
        self.allowed_num = allowed_num
        self._use_index = (use_index is not False) and ((not (allowed_num[0] == allowed_num[1] == 1)) or (use_index is True))
        self.allow_duplicates = allow_duplicates
        self.optional_for_loading = optional_for_loading
        self.help_name = help_name

        self.argument = Argument(name,
                                 default='',
                                 type=str,
                                 help='name of the %s class%s' % (help_name, '(es)' if self.use_index() else ''),
                                 registered=registered.names(),
                                 is_fixed=True)

    def use_index(self) -> bool:
        return self._use_index

    def is_optional(self) -> bool:
        return self.optional_for_loading

    def is_allowed_num(self, num: int) -> (bool, bool):
        """ returns (enough, not too many) arguments """
        return (num >= self.allowed_num[0]), (num <= self.allowed_num[1] or self.allowed_num[1] < 0)

    def limit_str(self) -> str:
        if self.allowed_num[0] == self.allowed_num[1]:
            return str(self.allowed_num[0])
        return "%d-%s" % (self.allowed_num[0], "n" if self.allowed_num[1] == -1 else str(self.allowed_num[1]))

    def get_remaining_options(self, used_options: [str] = None, sort=True) -> [str]:
        if used_options is None:
            return self.argument.registered
        if len(used_options) + 1 > self.allowed_num[1] and not self.allowed_num[1] < 0:
            return []
        if self.allow_duplicates:
            return self.argument.registered
        registered = self.argument.registered.copy()
        for n in used_options:
            registered.remove(n.split('#')[0])
        if sort:
            registered = sorted(registered, reverse=False)
        return registered

    def register(self, parser: ArgumentParser, registering_cls, index=None):
        """ register this meta argument with argparse """
        if not (self.allowed_num[0] == self.allowed_num[1] == 0):
            self.argument.register(parser, registering_cls, index)


class ArgsInterface:
    """
    enables subclasses to register (meta-) arguments for argument parsing
    """

    def __init__(self):
        pass

    def str(self) -> str:
        return '{cls}({dict})'.format(**{
            'cls': self.__class__.__name__,
            'dict': ', '.join(['%s: %s' % (k, str(v)) for k, v in self._str_dict().items()])
        })

    def _str_dict(self) -> dict:
        return {}

    @classmethod
    def extend_args(cls, args_list: [str]):
        """
        allow modifying the arguments list before other classes' arguments are dynamically added
        this should be used sparsely, as it is hard to keep track of
        """
        pass

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return []

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return []

    @classmethod
    def add_arguments(cls, parser: ArgumentParser, index=None):
        """ add arguments returned by 'args_to_add' """
        for arg in cls.args_to_add(index=index):
            arg.register(parser, cls, index)

    @classmethod
    def add_meta_arguments(cls, parser: ArgumentParser) -> [MetaArgument]:
        """ add meta-arguments returned by 'meta_args_to_add' and return them """
        meta_args = cls.meta_args_to_add()
        for m_arg in meta_args:
            m_arg.register(parser, cls, index=None)
        return meta_args

    @classmethod
    def sanitize_args(cls, args: Namespace) -> Namespace:
        """ sanitize arguments to avoid e.g. having " left in a string and applies parsing rules """
        assert isinstance(args, Namespace)
        for k, arg in Argument.all.items():
            args.__dict__[k] = arg.value(args)
        return args

    @classmethod
    def _parsed_meta_argument(cls, register_dict: RegisterDict, meta_name: str, args: args_type, index=None):
        """ get a class back """
        try:
            name = items(args)[meta_name]
            return register_dict.get(name)
        except KeyError:
            raise KeyError('Meta value "%s" not in args' % meta_name)

    @classmethod
    def _parsed_meta_arguments(cls, register_dict: RegisterDict, meta_name: str, args: args_type, index=None) -> [type]:
        """ get a list of classes back """
        try:
            names = items(args)[meta_name]
            return [register_dict.get(n) for n in split(names)]
        except KeyError:
            raise KeyError('Value "%s" not in args' % meta_name)

    @classmethod
    def _parsed_argument(cls, name: str, args: args_type, index=None, split_=None):
        """
        get an argument back, which had the class name added
        - does not work with fixed args (which are only intended for debugging anyway)
        - lists can be split
        """
        n = _arg_name(cls, name, index)
        try:
            arg = items(args)[n]
            if split_ is not None:
                arg = split(arg, cast_fun=split_)
            return arg
        except KeyError:
            raise KeyError('Value "%s" not in args, maybe it is not available due to missing pip-modules or a typo?' % n)

    @classmethod
    def _parsed_arguments(cls, names: [str], args: args_type, index=None) -> list:
        """ get all argument back, which had the class name added """
        return [cls._parsed_argument(n, args, index) for n in names]

    @classmethod
    def _all_parsed_arguments(cls, args: args_type, index=None) -> dict:
        """ gets a dictionary of all {names: values} that this class registered via argparse """
        return {n.name: cls._parsed_argument(name=n.name, args=args, index=index) for n in cls.args_to_add(index=index)}

    @classmethod
    def init_multiple(cls, register_dict: RegisterDict, args: args_type, split_key: str) -> list:
        """ creates list of ArgsInterface objects """
        splits = split(items(args)[split_key])
        return [register_dict.get(cls_name)(args, i) for i, cls_name in enumerate(splits)]

    @classmethod
    def parsed_argument_defaults(cls) -> dict:
        """ dict of {name: default value} all arguments """
        return {arg.name: arg.apply_rules(arg.default) for arg in cls.args_to_add(index=None)}

    @classmethod
    def matches_registered_properties(cls, **kwargs) -> bool:
        """ return whether this class was registered with the specified properties """
        registered_kwargs = Register.get_my_kwargs(cls)
        for k, v in kwargs.items():
            if k not in registered_kwargs:
                raise ValueError("Class %s has no registered property %s" % (cls.__name__, k))
            if not v == registered_kwargs.get(k):
                return False
        return True


class ArgsTreeNode:
    """
    build and/or parse the argparse namespace
    """

    def __init__(self, args_cls: ArgsInterface.__class__, depth=1, _in_meta: MetaArgument = None, _index=None):
        self.args_cls = args_cls            # class this node belongs to
        self.children = defaultdict(list)   # {meta name: list of ArgsTreeNode}
        self.metas = dict()                 # {meta name: MetaArgument}
        self.depth = depth                  # tree node depth
        self._in_meta = _in_meta            # under which meta argument name this node is created
        self._is_root = _in_meta is None    # only the root node is not registered within a meta argument
        self.index = _index                 # if there are multiple classes in one meta argument, they are indexed

    def reset(self):
        """ remove all children """
        self.children = defaultdict(list)

    def _print(self, *text):
        print('. ' * self.depth, self.name, *text)

    @property
    def name(self) -> str:
        if self.index is None:
            return self.args_cls.__name__
        return '%s#%d' % (self.args_cls.__name__, self.index)

    def build_from_args(self, args_list: [str], parser: ArgumentParser = None, raise_problems=True):
        """
        recursively parse known arguments and let child nodes parse theirs

        :param parser:
        :param args_list:
        :param raise_problems:
        :return:
        """
        if parser is None:
            parser = ArgumentParser('tmp parser')
        self.args_cls.extend_args(args_list=args_list)
        meta_args = self.args_cls.add_meta_arguments(parser)
        if len(meta_args) > 0:
            args, _ = parser.parse_known_args(args_list)
            args = self.args_cls.sanitize_args(args)
            for meta in meta_args:
                self.metas[meta.argument.name] = meta
                meta_name = meta.argument.name
                cls_names = split(items(args)[meta_name])
                self._print('has meta argument:', meta_name, cls_names)
                if len(cls_names) == 0 and meta.optional_for_loading:
                    self._print('missing optional values for', meta_name)
                elif raise_problems and not meta.is_allowed_num(len(cls_names)):
                    raise ValueError('Only %s classed allowed for %s, have %s'
                                     % (meta.allowed_num, meta_name, str(cls_names)))
                for cls_name in cls_names:
                    self.add_child_meta(meta, cls_name)
                for child in self.children[meta.argument.name]:
                    child.build_from_args(args_list, parser=parser, raise_problems=raise_problems)

    def _add_child(self, meta: MetaArgument, child):
        self.children[meta.argument.name].append(child)

    def add_child_meta(self, meta: MetaArgument, cls_name: str):
        # add new node to children
        assert cls_name in meta.argument.registered,\
            "Can not add %s for %s! " \
            "\nThe class may not available in this context (does not make sense)," \
            "\nor at all (e.g. an optional class is not loaded if the respective python libraries are missing)"\
            % (cls_name, meta.argument.name)
        cls = meta.registered.get(cls_name)
        c = self.__class__(cls, self.depth + 1, _in_meta=meta, _index=None)
        _, a2 = meta.is_allowed_num(len(self.children[meta.argument.name])+1)
        if not a2:
            raise ValueError('Too many classes for %s, allowed: %s' % (meta.argument.name, str(meta.allowed_num)))
        if not meta.allow_duplicates:
            for c2 in self.children[meta.argument.name]:
                if c.args_cls.__name__ == c2.args_cls.__name__:
                    raise ValueError('Duplicate class "%s", only allowed once!' % c.args_cls.__name__)
        self._add_child(meta, c)
        self.metas[meta.argument.name] = meta
        # update indices of children
        self._update_indices(meta)

    def _update_indices(self, meta: MetaArgument):
        if meta.use_index():
            for i, c in enumerate(self.children[meta.argument.name]):
                c.index = i

    def _can_parse(self, args_list: [str], parser: ArgumentParser):
        """ enable the gui to update current values, not needed for pure parsing """
        pass

    def get_wildcards(self) -> dict:
        """ get wildcards of this node """
        wildcards = {}
        if isinstance(self._in_meta, MetaArgument):
            fmt = '%s#%d' % ('%s', self.index) if self.index is not None else '%s'
            wildcards[fmt % self._in_meta.argument.name] = fmt % self.args_cls.__name__
        return wildcards

    def parse(self, args_list: [str], parser: ArgumentParser = None, raise_unparsed=True) -> (Namespace, dict, list, dict):
        """
        parse the list of arguments

        :param args_list:
        :param parser: optional ArgumentParser
        :param raise_unparsed: raise an exception when there are unparsed arguments
        :return:
            if parser is None: (None, wildcards, None, description each arg)
            else: (Namespace of the arguments, wildcards, list of unparsed arguments, description each arg)
        """
        # add wildcard
        self._print('parsing')
        wildcards = self.get_wildcards()
        descriptions = {}
        for meta_name, meta in self.metas.items():
            descriptions[meta.argument.name] = meta.help_name
        # add arguments of this node
        if parser is not None:
            for arg in self.args_cls.args_to_add(index=self.index):
                arg.register(parser, self.args_cls, self.index)
                descriptions['%s.%s' % (self.name, arg.name)] = arg.help
            self._can_parse(args_list, parser)
        for meta_name, children in self.children.items():
            for child in children:
                _, w, _, d = child.parse(args_list, parser, raise_unparsed=False)
                wildcards.update(w)
                descriptions.update(d)
        # the root node finally parses
        if self._is_root:
            # update wildcards with commonly used paths
            args_list, failed_args = replace_wildcards_in_args_list(args_list, wildcards)
            # parse
            if parser is not None:
                args, unparsed = parser.parse_known_args(args=args_list)
                args = self.args_cls.sanitize_args(args)
                if len(unparsed) > 0:
                    if raise_unparsed:
                        raise ValueError('Unparsed arguments! %s' % ', '.join(unparsed))
                    print('Unparsed arguments!', unparsed)
                return args, wildcards, failed_args, descriptions
            return None, wildcards, None, descriptions
        return None, wildcards, None, descriptions
