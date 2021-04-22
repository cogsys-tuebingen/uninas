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

    def filter_match_all(self, **kwargs) -> 'RegisterDict':
        """ subset that contains only the items that match all of the kwargs """
        return self._filter_match(all, **kwargs)

    def filter_match_any(self, **kwargs) -> 'RegisterDict':
        """ subset that contains the items that match any of the kwargs """
        return self._filter_match(any, **kwargs)

    def get_item(self, key: str) -> RegisteredItem:
        """
        get a registered item
        :param key: under which key/name it registered with (e.g. class name)
        """
        if key is None:
            raise NotImplementedError('Key is None')
        used_key = str(key).lower().replace('"', '').replace("'", '')
        for key, item in self.items():
            if key == used_key:
                return item
        raise ModuleNotFoundError('Unregistered key "%s"!' % key)

    def get(self, key: str):
        """
        get a registered item value
        :param key: under which key/name it registered with (e.g. class name)
        """
        return self.get_item(key).value


class Register(metaclass=Singleton):
    builder = None  # is set immediately
    _missing_imports = []

    all = RegisterDict('__all__', add_to_all=False)
    tasks = RegisterDict('tasks')
    trainers = RegisterDict('trainer')
    training_callbacks = RegisterDict('training callbacks')
    training_clones = RegisterDict('training clones')
    exp_loggers = RegisterDict('experiment loggers')
    devices_managers = RegisterDict('device managers')
    data_sets = RegisterDict('data sets')
    benchmark_sets = RegisterDict('benchmark sets')
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
    network_mixed_ops = RegisterDict('network mixed ops')
    network_layers = RegisterDict('network layers')
    network_blocks = RegisterDict('network blocks')
    network_cells = RegisterDict('network cells')
    network_bodies = RegisterDict('network bodies')
    networks = RegisterDict('networks')
    models = RegisterDict('models')
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
    nas_metrics = RegisterDict('NAS metrics')

    @classmethod
    def _add(cls, dct: RegisterDict, **kwargs):
        """
        returns a decorator that registers the decorated callable and remembers the kwargs
        """
        def _decorator(register_callable: Callable):
            # the main thread may register items before the builder can do so (due to imports), skip all of them
            if '__main__' in repr(register_callable):
                return register_callable
            # otherwise make sure that nothing can be registered with the same key/name
            key = register_callable.__name__
            lower_key = key.lower()
            if lower_key in cls.all:
                assert False, 'The key "%s" can not be registered multiple times!' % key
            item = RegisteredItem(key, register_callable, kwargs)
            cls.all[lower_key] = item
            dct[lower_key] = item
            return register_callable

        return _decorator

    @classmethod
    def task(cls, search=False):
        """
        register a task
        :param search: if the task is about searching a network architecture
        """
        return cls._add(cls.tasks, search=search)

    @classmethod
    def trainer(cls):
        """
        register a trainer
        """
        return cls._add(cls.trainers)

    @classmethod
    def training_callback(cls, requires_log_dict=False):
        """
        register a training callback
        :param requires_log_dict: the callback requires the log_dict (containing e.g. evaluated metric info) to function
        """
        return cls._add(cls.training_callbacks, requires_log_dict=requires_log_dict)

    @classmethod
    def training_clone(cls):
        """
        register a training clone
        """
        return cls._add(cls.training_clones)

    @classmethod
    def exp_logger(cls):
        """
        register a lightning experiment logger
        """
        return cls._add(cls.exp_loggers)

    @classmethod
    def devices_manager(cls):
        """
        register device manager (e.g. cpu, cuda)
        """
        return cls._add(cls.devices_managers)

    @classmethod
    def data_set(cls, images=False, classification=False, regression=False):
        """
        register a data set
        :param images: the data set contains images
        :param classification: the data set is for classification
        :param regression: the data set is for regression
        """
        return cls._add(cls.data_sets, images=images, classification=classification, regression=regression)

    @classmethod
    def benchmark_set(cls, mini=False, tabular=False, surrogate=False):
        """
        register a benchmark set for architecture benchmarking
        :param mini: only contains final training results
        :param tabular: has a lookup table of trained architectures / results
        :param surrogate: creates results through some model
        """
        return cls._add(cls.benchmark_sets, mini=mini, tabular=tabular, surrogate=surrogate)

    @classmethod
    def augmentation_set(cls, on_single=False, on_batch=False, on_images=False):
        """
        register a set of data augmentations
        :param on_single: augments a single data point (e.g. image)
        :param on_batch: augments a batch (e.g. mixup)
        :param on_images: specifically designed for images, e.g. makes use of PIL.Image
        """
        return cls._add(cls.augmentation_sets, on_single=on_single, on_batch=on_batch, on_images=on_images)

    @classmethod
    def criterion(cls, only_head=False, distill=False):
        """
        register a training criterion
        :param only_head: can only be used on the network output
        :param distill: the criterion compares student network to teacher network (not to target)
        """
        return cls._add(cls.criteria, only_head=only_head, distill=distill)

    @classmethod
    def metric(cls, only_head=False, distill=False):
        """
        register a metric
        :param only_head: if the metric is only computed on the head outputs
        :param distill: if the metric is used to distill (intermediate) network features for another net,
                        and does not use the true labels
        """
        return cls._add(cls.metrics, only_head=only_head, distill=distill)

    @classmethod
    def method(cls, search=False, single_path=False, distill=False, can_hpo=False):
        """
        register a method
        :param search: if the method is used to search for an architecture
        :param single_path: if the strategy in the method only ever returs a single path for each choice
        :param distill: if the method is used to search for an architecture by distilling a teacher
        :param can_hpo: if the method can be used for hyper-parameter optimization of the architecture
        """
        return cls._add(cls.methods, search=search, single_path=single_path, distill=distill, can_hpo=can_hpo)

    @classmethod
    def strategy(cls, single_path=False, can_hpo=False):
        """
        register a weight strategy for architecture search
        :param single_path: if the strategy only ever returs a single path for each choice
        :param can_hpo: if the strategy can be used for hyper-parameter optimization of the architecture
        """
        return cls._add(cls.strategies, single_path=single_path, can_hpo=can_hpo)

    @classmethod
    def primitive_set(cls):
        """
        register a set of primitives (different paths in architecture search)
        """
        return cls._add(cls.primitive_sets)

    @classmethod
    def attention_module(cls):
        """
        register a network attention module
        """
        return cls._add(cls.attention_modules)

    @classmethod
    def act_fun(cls):
        """
        register an activation function
        """
        return cls._add(cls.act_funs)

    @classmethod
    def network_stem(cls):
        """
        register a network stem
        """
        return cls._add(cls.network_stems)

    @classmethod
    def network_head(cls):
        """
        register a network head
        """
        return cls._add(cls.network_heads)

    @classmethod
    def network_module(cls):
        """
        register a network module
        """
        return cls._add(cls.network_modules)

    @classmethod
    def network_mixed_op(cls):
        """
        register a network mixed op
        """
        return cls._add(cls.network_mixed_ops)

    @classmethod
    def network_layer(cls):
        """
        register a network layer
        """
        return cls._add(cls.network_layers)

    @classmethod
    def network_block(cls):
        """
        register a network block
        """
        return cls._add(cls.network_blocks)

    @classmethod
    def network_cell(cls):
        """
        register a network cell
        """
        return cls._add(cls.network_cells)

    @classmethod
    def network_body(cls):
        """
        register a network body
        """
        return cls._add(cls.network_bodies)

    @classmethod
    def network(cls, search=False, only_config=False, external=False):
        """
        register a network (a specific kind of model)
        :param search: if the network supports architecture search
        :param only_config: if the network can be built from a config file alone
        :param external: if the network is from an external source
        """
        return cls._add(cls.networks, search=search, only_config=only_config, external=external)

    @classmethod
    def model(cls, can_fit=False, classification=False, regression=False):
        """
        register a model
        :param can_fit: the model can be directly fit (no trainer needed)
        :param classification: the model is for classification
        :param regression: the model is for regression
        """
        return cls._add(cls.models, can_fit=can_fit, classification=classification, regression=regression)

    @classmethod
    def initializer(cls):
        """
        register a network initializer
        """
        return cls._add(cls.initializers)

    @classmethod
    def regularizer(cls):
        """
        register a network regularizer
        """
        return cls._add(cls.regularizers)

    @classmethod
    def optimizer(cls):
        """
        register an optimizer
        """
        return cls._add(cls.optimizers)

    @classmethod
    def scheduler(cls):
        """
        register a lr scheduler
        """
        return cls._add(cls.schedulers)

    @classmethod
    def hpo_estimator(cls, requires_trainer=False, requires_method=False, requires_bench=False):
        """
        register a hpo estimator
        :param requires_trainer: if the estimator requires a trainer (e.g. for forward passes to compute the loss)
        :param requires_method: if the estimator requires a method (e.g. to have access to the network)
        :param requires_bench: if the estimator requires a benchmark
        """
        return cls._add(cls.hpo_estimators, requires_trainer=requires_trainer, requires_method=requires_method,
                        requires_bench=requires_bench)

    @classmethod
    def hpo_self_algorithm(cls):
        """
        register a self hpo algorithm
        """
        return cls._add(cls.hpo_self_algorithms)

    @classmethod
    def hpo_pymoo_terminator(cls):
        """
        register a pymoo terminator class
        """
        return cls._add(cls.hpo_pymoo_terminators)

    @classmethod
    def hpo_pymoo_sampler(cls):
        """
        register a pymoo sampler class
        """
        return cls._add(cls.hpo_pymoo_samplers)

    @classmethod
    def hpo_pymoo_crossover(cls):
        """
        register a pymoo crossover class
        """
        return cls._add(cls.hpo_pymoo_crossovers)

    @classmethod
    def hpo_pymoo_mutation(cls):
        """
        register a pymoo mutation
        """
        return cls._add(cls.hpo_pymoo_mutations)

    @classmethod
    def hpo_pymoo_algorithm(cls):
        """
        register a pymoo algorithm
        """
        return cls._add(cls.hpo_pymoo_algorithms)

    @classmethod
    def pbt_selector(cls):
        """
        register a pbt selector
        """
        return cls._add(cls.pbt_selectors)

    @classmethod
    def pbt_mutation(cls):
        """
        register a pbt mutation
        """
        return cls._add(cls.pbt_mutations)

    @classmethod
    def optimization_target(cls):
        """
        register an optimization target
        """
        return cls._add(cls.optimization_targets)

    @classmethod
    def profiler(cls):
        """
        register a profiler
        """
        return cls._add(cls.profilers)

    @classmethod
    def profile_function(cls):
        """
        register a profile function
        """
        return cls._add(cls.profile_functions)

    @classmethod
    def nas_metric(cls, is_correlation=False):
        """
        register a NAS metric
        :param is_correlation: a correlation metric
        """
        return cls._add(cls.nas_metrics, is_correlation=is_correlation)

    @classmethod
    def _print_dct(cls, logger: Logger, text: str, dct: RegisterDict):
        logger.info('{:<25}{}'.format(text, str(dct.names())))

    @classmethod
    def log_all(cls, logger: Logger):
        """
        log all registered items
        :param logger: where to log to
        """
        log_headline(logger, 'Registered classes')
        log_in_columns(logger, [(dct.text, str(dct.names())) for dct in RegisterDict.all], add_bullets=True)

    @classmethod
    def get_item(cls, key: str) -> RegisteredItem:
        """
        get a registered item
        :param key: under which key/name it registered with (e.g. class name)
        """
        return cls.all.get_item(key)

    @classmethod
    def get(cls, key: str):
        """
        get a registered item value
        :param key: under which key/name it registered with (e.g. class name)
        """
        return cls.get_item(key).value

    @classmethod
    def get_my_kwargs(cls, cls_: type):
        """
        get the kwargs that the class registered with
        :param cls_: class
        """
        return cls.get_item(cls_.__name__).kwargs

    @classmethod
    def missing_import(cls, e: ImportError):
        """
        if an optional import is missing, note it here
        :param e:
        """
        if e.name in cls._missing_imports:
            return
        cls._missing_imports.append(e.name)
        print("Missing module, some features are not available (%s)" % repr(e))
