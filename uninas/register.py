"""
networks, strategies, data sets, optimizers, ... can register themselves here
- easy access via name from anywhere
- filter groups by kwargs
"""


from typing import Callable
from uninas.utils.meta import Singleton
from uninas.utils.loggers.python import Logger, log_in_columns, log_headline


class RegisteredItem:
    def __init__(self, name: str, value: Callable, kwargs: dict):
        self.name = name
        self.value = value
        self.kwargs = kwargs


class RegisterDict(dict):
    all = []  # list of all RegisterDicts

    def __init__(self, text: str, add_to_all=True):
        super().__init__()
        self.text = text
        if add_to_all:
            self.all.append(self)

    def names(self) -> [str]:
        """ list the names of all items """
        return sorted([item.name for item in self.values()], reverse=False)

    def _filter_match(self, fun, **kwargs):
        c = RegisterDict(self.text, add_to_all=False)
        for k, v in self.items():
            if fun([v.kwargs.get(k2, None) == v2 for k2, v2 in kwargs.items()]):
                c[k] = v
        return c

    def filter_match_all(self, **kwargs):
        """ subset that contains only the items that match all of the kwargs """
        return self._filter_match(all, **kwargs)

    def filter_match_any(self, **kwargs):
        """ subset that contains the items that match any of the kwargs """
        return self._filter_match(any, **kwargs)


class Register(metaclass=Singleton):
    builder = None  # is set immediately
    _missing_imports = []

    all = RegisterDict('__all__', add_to_all=False)
    tasks = RegisterDict('tasks')
    trainers = RegisterDict('trainer')
    training_callbacks = RegisterDict('training callbacks')
    exp_loggers = RegisterDict('experiment loggers')
    devices_managers = RegisterDict('device managers')
    data_sets = RegisterDict('data sets')
    augmentation_sets = RegisterDict('augmentation sets')
    criteria = RegisterDict('criteria')
    metrics = RegisterDict('metrics')
    methods = RegisterDict('methods')
    strategies = RegisterDict('strategies')
    primitive_sets = RegisterDict('primitive sets')
    attention_modules = RegisterDict('attention modules')
    act_funs = RegisterDict('activation functions')
    network_stems = RegisterDict('network stems')
    network_heads = RegisterDict('network heads')
    network_modules = RegisterDict('network modules')
    network_layers = RegisterDict('network layers')
    network_blocks = RegisterDict('network blocks')
    network_cells = RegisterDict('network cells')
    network_bodies = RegisterDict('network bodies')
    networks = RegisterDict('networks')
    initializers = RegisterDict('initializers')
    regularizers = RegisterDict('regularizers')
    optimizers = RegisterDict('optimizers')
    schedulers = RegisterDict('schedulers')
    hpo_estimators = RegisterDict('hpo estimators')
    hpo_self_algorithms = RegisterDict('hpo self algorithms')
    hpo_pymoo_terminators = RegisterDict('hpo pymoo algorithms')
    hpo_pymoo_samplers = RegisterDict('hpo pymoo terminators')
    hpo_pymoo_crossovers = RegisterDict('hpo pymoo samplers')
    hpo_pymoo_mutations = RegisterDict('hpo pymoo crossovers')
    hpo_pymoo_algorithms = RegisterDict('hpo pymoo mutations')
    pbt_selectors = RegisterDict('pbt selectors')
    pbt_mutations = RegisterDict('pbt mutations')
    optimization_targets = RegisterDict('optimization targets')
    profilers = RegisterDict('profiler')
    profile_functions = RegisterDict('profile functions')
    correlation_metrics = RegisterDict('correlation metrics')

    @classmethod
    def _add(cls, dct: dict, **kwargs):
        """
        returns a decorator that registers the decorated callable and remembers the kwargs
        """
        def _decorator(register_callable: Callable):
            key = register_callable.__name__
            lower_key = key.lower()
            if lower_key in cls.all:
                assert register_callable == cls.all[lower_key],\
                    'The key "%s" can not be registered multiple times!' % key
            item = RegisteredItem(key, register_callable, kwargs)
            cls.all[lower_key] = item
            dct[lower_key] = item
            return register_callable
        return _decorator

    @classmethod
    def task(cls, search=False):
        return cls._add(cls.tasks, search=search)

    @classmethod
    def trainer(cls):
        return cls._add(cls.trainers)

    @classmethod
    def training_callback(cls, requires_log_dict=False):
        return cls._add(cls.training_callbacks, requires_log_dict=requires_log_dict)

    @classmethod
    def exp_logger(cls):
        return cls._add(cls.exp_loggers)

    @classmethod
    def devices_manager(cls):
        return cls._add(cls.devices_managers)

    @classmethod
    def data_set(cls, images=False):
        return cls._add(cls.data_sets, images=images)

    @classmethod
    def augmentation_set(cls, on_single=False, on_batch=False, on_images=False):
        return cls._add(cls.augmentation_sets, on_single=on_single, on_batch=on_batch, on_images=on_images)

    @classmethod
    def criterion(cls, only_head=False, distill=False):
        return cls._add(cls.criteria, only_head=only_head, distill=distill)

    @classmethod
    def metric(cls, only_head=False, distill=False):
        return cls._add(cls.metrics, only_head=only_head, distill=distill)

    @classmethod
    def method(cls, search=False, single_path=False, distill=False, can_hpo=False):
        return cls._add(cls.methods, search=search, single_path=single_path, distill=distill, can_hpo=can_hpo)

    @classmethod
    def strategy(cls, single_path=False, can_hpo=False):
        return cls._add(cls.strategies, single_path=single_path, can_hpo=can_hpo)

    @classmethod
    def primitive_set(cls):
        return cls._add(cls.primitive_sets)

    @classmethod
    def attention_module(cls):
        return cls._add(cls.attention_modules)

    @classmethod
    def act_fun(cls):
        return cls._add(cls.act_funs)

    @classmethod
    def network_stem(cls):
        return cls._add(cls.network_stems)

    @classmethod
    def network_head(cls):
        return cls._add(cls.network_heads)

    @classmethod
    def network_module(cls):
        return cls._add(cls.network_modules)

    @classmethod
    def network_layer(cls):
        return cls._add(cls.network_layers)

    @classmethod
    def network_block(cls):
        return cls._add(cls.network_blocks)

    @classmethod
    def network_cell(cls):
        return cls._add(cls.network_cells)

    @classmethod
    def network_body(cls):
        return cls._add(cls.network_bodies)

    @classmethod
    def network(cls, search=False, only_config=False, external=False):
        return cls._add(cls.networks, search=search, only_config=only_config, external=external)

    @classmethod
    def initializer(cls):
        return cls._add(cls.initializers)

    @classmethod
    def regularizer(cls):
        return cls._add(cls.regularizers)

    @classmethod
    def optimizer(cls):
        return cls._add(cls.optimizers)

    @classmethod
    def scheduler(cls):
        return cls._add(cls.schedulers)

    @classmethod
    def hpo_estimator(cls):
        return cls._add(cls.hpo_estimators)

    @classmethod
    def hpo_self_algorithm(cls):
        return cls._add(cls.hpo_self_algorithms)

    @classmethod
    def hpo_pymoo_terminator(cls):
        return cls._add(cls.hpo_pymoo_terminators)

    @classmethod
    def hpo_pymoo_sampler(cls):
        return cls._add(cls.hpo_pymoo_samplers)

    @classmethod
    def hpo_pymoo_crossover(cls):
        return cls._add(cls.hpo_pymoo_crossovers)

    @classmethod
    def hpo_pymoo_mutation(cls):
        return cls._add(cls.hpo_pymoo_mutations)

    @classmethod
    def hpo_pymoo_algorithm(cls):
        return cls._add(cls.hpo_pymoo_algorithms)

    @classmethod
    def pbt_selector(cls):
        return cls._add(cls.pbt_selectors)

    @classmethod
    def pbt_mutation(cls):
        return cls._add(cls.pbt_mutations)

    @classmethod
    def optimization_target(cls):
        return cls._add(cls.optimization_targets)

    @classmethod
    def profiler(cls):
        return cls._add(cls.profilers)

    @classmethod
    def profile_function(cls):
        return cls._add(cls.profile_functions)

    @classmethod
    def correlation_metric(cls, rank=False):
        return cls._add(cls.correlation_metrics, rank=rank)

    @classmethod
    def _print_dct(cls, logger: Logger, text: str, dct: RegisterDict):
        logger.info('{:<25}{}'.format(text, str(dct.names())))

    @classmethod
    def log_all(cls, logger: Logger):
        log_headline(logger, 'Registered classes')
        log_in_columns(logger, [(dct.text, str(dct.names())) for dct in RegisterDict.all], add_bullets=True)

    @classmethod
    def get_item(cls, key: str) -> RegisteredItem:
        if key is None:
            raise NotImplementedError('Key is None')
        used_key = str(key).lower().replace('"', '').replace("'", '')
        if used_key not in cls.all:
            raise ModuleNotFoundError('Unregistered key "%s"!' % key)
        return cls.all[used_key]

    @classmethod
    def get(cls, key: str):
        return cls.get_item(key).value

    @classmethod
    def get_my_kwargs(cls, cls_: type):
        """ get the kwargs that the class registered with """
        return cls.get_item(cls_.__name__).kwargs

    @classmethod
    def missing_import(cls, e: ImportError):
        if e.name in cls._missing_imports:
            return
        cls._missing_imports.append(e.name)
        print("Missing module %s, some features are not available." % e.name)
