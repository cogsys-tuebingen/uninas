import torch
from typing import Union, List
from torch import nn
from uninas.optimization.hpo.uninas.values import ValueSpace, DiscreteValues
from uninas.methods.abstract_strategy import AbstractWeightStrategy
from uninas.utils.loggers.python import Logger, log_headline, log_in_columns
from uninas.utils.meta import Singleton


class StrategyManager(nn.Module, metaclass=Singleton):
    """
    Keeping track of (multiple) WeightStrategies
    """

    def __init__(self):
        super().__init__()
        self.strategies = nn.ModuleDict()   # {strategy name: strategy}
        self._weight_to_strategy = {}       # {weight name: strategy name}
        self._ordered = []                  # weight names in order of requests
        self._ordered_unique = []           # weight names in order of requests, ignoring duplicates
        self._fixed_strategy_name = None    # force all weights to be registered under this strategy (name)

    def reset(self):
        self.strategies.clear()
        self._weight_to_strategy.clear()
        self._ordered.clear()
        self._ordered_unique.clear()

    def set_fixed_strategy_name(self, name: str = None):
        """
        force all weights to be registered under this strategy (name)
        mostly useful for loading a (trained) super-network that requires multiple weight strategies from a config
        """
        self._fixed_strategy_name = name

    def get_fixed_strategy_name(self) -> Union[str, None]:
        return self._fixed_strategy_name

    def _strategy_name(self, name: str) -> str:
        if isinstance(self._fixed_strategy_name, str):
            return self._fixed_strategy_name
        return name

    def add_strategy(self, strategy: AbstractWeightStrategy) -> 'StrategyManager':
        """
        add a weight strategy to the manager
        """
        if strategy.name in self.strategies:
            raise KeyError('strategy name %s already in use' % strategy.name)
        self.strategies[strategy.name] = strategy
        return self

    def delete_strategy(self, name: str) -> 'StrategyManager':
        """
        delete a weight strategy from the manager
        """
        strategy = self.strategies[name] if name in self.strategies else None
        if self._fixed_strategy_name == name:
            self._fixed_strategy_name = None
        if isinstance(strategy, AbstractWeightStrategy):
            for n in strategy.get_weight_names():
                self._weight_to_strategy.pop(n)
                self._ordered.remove(n)
                self._ordered_unique.remove(n)
            del self.strategies[name]
            del strategy
        return self

    def log_detailed(self, logger: Logger):
        def _get_name_rows(s: Union[StrategyManager, AbstractWeightStrategy], prefix: str) -> list:
            order1, order2 = s.ordered_names(unique=False), s.ordered_names(unique=True)
            if len(order1) == len(order2):
                return [
                    ('%s weights in request order (all are unique):' % prefix, str(order1)),
                ]
            return [
                ('%s weights in request order (not unique):' % prefix, str(order1)),
                ('%s weights in request order (unique):' % prefix, str(order2)),
            ]

        txt = "Weight strategies" if len(self.strategies) > 1 else "Weight strategy"
        log_headline(logger, txt)
        strategies = self.get_strategies_list()
        if len(strategies) > 1:
            rows = _get_name_rows(self, 'all')
            log_in_columns(logger, rows)
            logger.info("")

        for i, strategy in enumerate(strategies):
            assert isinstance(strategy, AbstractWeightStrategy)
            if i > 0:
                logger.info("")

            logger.info(strategy.str())
            rows = [("name", "num choices", "used")]
            for r in strategy.get_requested_weights():
                rows.append((r.name, r.num_choices_str(), '%dx' % r.num_requests()))
            logger.info("Weights:")
            log_in_columns(logger, rows, add_bullets=True, num_headers=1)

            rows = _get_name_rows(strategy, 'strategy')
            log_in_columns(logger, rows)

    def get_strategies(self) -> {str: AbstractWeightStrategy}:
        return self.strategies

    def get_strategies_list(self) -> [AbstractWeightStrategy]:
        return list(sorted(self.strategies.values(), key=lambda s: s.name))

    def get_strategy_by_weight(self, weight_name: str) -> Union[AbstractWeightStrategy, None]:
        strategy_name = self._weight_to_strategy.get(weight_name, None)
        if strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            assert isinstance(strategy, AbstractWeightStrategy)
            return strategy
        return None

    def is_only_single_path(self) -> bool:
        """ whether all used strategies are single-path """
        return all([strategy.is_single_path() for strategy in self.get_strategies().values()])

    def uses_all_paths(self) -> bool:
        """ whether a forward pass will use all parameters (used for distributed training) """
        for strategy in self.get_strategies().values():
            if strategy.is_single_path():
                return False
        return True

    def make_weight(self, strategy_name: str, name: str, only_single_path=False,
                    choices: nn.ModuleList = None, num_choices: int = None) -> AbstractWeightStrategy:
        """
        register that a parameter of given name and num choices will be required during the search
        called by network components before the network is built

        :param strategy_name: name of the strategy to register the weight with
        :param name: name of the weight (may be shared by multiple modules)
        :param only_single_path: make sure that the weight strategy is single-path (can make the implementation easier)
        :param choices: module list of the options
        :param num_choices: number of options, required if 'choices' is None
        """
        strategy_name = self._strategy_name(strategy_name)
        ws1 = self.strategies[strategy_name] if strategy_name in self.strategies else None
        ws2 = self.get_strategy_by_weight(weight_name=name)
        num_choices = len(choices) if choices is not None else num_choices

        if ws1 is None:
            raise KeyError('strategy with name "%s" does not exist' % strategy_name)
        assert isinstance(ws1, AbstractWeightStrategy), "strategy '%s' is actually not a strategy" % strategy_name
        if num_choices < 1:
            raise ValueError('can not have a weight without option(s) to choose from')
        if ws2 not in [ws1, None]:
            raise ValueError('can not register the same weight name in different strategies')

        if name not in self._ordered:
            self._ordered_unique.append(name)
        self._ordered.append(name)
        self._weight_to_strategy[name] = strategy_name
        if only_single_path:
            assert ws1.is_single_path(),\
                "The network module requires to use a single architecture path, " \
                "but the strategy %s may return multiple" % ws1.__class__.__name__
        ws1.make_weight(name=name, choices=choices, num_choices=num_choices)
        return ws1

    def mask_index(self, idx: int, weight_name: str = None):
        """ prevent sampling a specific index, either for a specific weight or all weights """
        if isinstance(weight_name, str):
            strategies = [self.get_strategy_by_weight(weight_name)]
        else:
            strategies = self.strategies.values()
        for s in strategies:
            s.mask_index(idx, weight_name=weight_name)

    def build(self):
        """
        build all weight strategies
        """
        for ws in self.strategies.values():
            ws.build()

    def forward_const(self, const=0):
        """ set all arc weights to a constant value """
        for ws in self.strategies.values():
            ws.forward_const(const=const)

    def forward(self, fixed_arc: [int] = None, strategy_dict: dict = None):
        """
        forward pass each WeightStrategy
        :param fixed_arc: optional list of unique indices to fix the architecture, overwrites any fixed_arc in name_to_dict
        :param strategy_dict: {strategy name: strategy specific stuff}
        """
        if isinstance(fixed_arc, (tuple, list)):
            strategy_dict = {} if strategy_dict is None else strategy_dict
            # each strategy gets only the indices that belong to its weights
            for s in self.strategies.values():
                strategy_dict[s.name] = strategy_dict.get(s.name, {})
                strategy_dict[s.name]['fixed_arc'] = []
            for n, v in zip(self._ordered_unique, fixed_arc):
                strategy_dict[self.get_strategy_by_weight(n).get_name()]['fixed_arc'].append(v)
        # if a strategy dict is given, execute only these strategies
        if strategy_dict is not None:
            for k, v in strategy_dict.items():
                self.strategies[k].forward(**v)
        else:
            for n, s in self.strategies.items():
                s.forward()

    def get_all_finalized_indices(self, unique=True, flat=False) -> Union[List[int], List[List[int]]]:
        """
        :param flat: if True, return only one integer per index, will fail for multi-path strategies
                     if False return a list of indices instead
        :param unique: False to consider weights that are used multiple times also multiple times in the indices
        :return: indices of the modules that should constitute the new architecture
                 a list of only ints if flat, otherwise a list of indices at every position
        """
        if flat:
            return [self.get_strategy_by_weight(n).get_finalized_index(n) for n in self.ordered_names(unique=unique)]
        return [self.get_strategy_by_weight(n).get_finalized_indices(n) for n in self.ordered_names(unique=unique)]

    def get_finalized_indices(self, weight_name: str, flat=False) -> Union[int, List[int]]:
        """
        it's slightly more efficient to directly call this on the specific AbstractStrategy that is returned
        when creating a weight

        :param weight_name: name of the weight
        :param flat: if True, return only one integer, will fail for multi-path strategies
                     if False return a list of indices instead
        :return: indices of the modules that should constitute the new architecture
                 a list of only ints if flat, otherwise a list of indices at every position
        """
        if flat:
            return self.get_strategy_by_weight(weight_name).get_finalized_index(weight_name)
        return self.get_strategy_by_weight(weight_name).get_finalized_indices(weight_name)

    def ordered_names(self, unique=True) -> [str]:
        """
        name of each weight, by request order
        """
        lst = self._ordered_unique if unique else self._ordered
        return lst.copy()

    def get_log_dict(self) -> {str: float}:
        """
        :return: dict of values that are interesting to log
        """
        dct = {}
        for i, ws in enumerate(self.strategies.values()):
            for k, v in ws.get_log_dict().items():
                dct["ws%d/%s" % (i, k)] = v
        return dct

    def get_losses(self, clear=True) -> {str, torch.Tensor}:
        """ get loss tensors, maybe clear storage """
        dct = {}
        for i, ws in enumerate(self.strategies.values()):
            for k, loss in ws.get_losses(clear=clear).items():
                dct["ws%d/%s" % (i, k)] = loss.unsqueeze(dim=0)
        return dct

    def on_epoch_start(self, current_epoch: int):
        """ whenever the method starts a new epoch """
        for ws in self.strategies.values():
            ws.on_epoch_start(current_epoch=current_epoch)

    def on_epoch_end(self, current_epoch: int) -> bool:
        """
        whenever the method ends an epoch
        signal early stopping when returning True
        """
        return any([ws.on_epoch_end(current_epoch=current_epoch) for ws in self.strategies.values()])

    def max_num_choices(self) -> int:
        return max([ws.max_num_choices() for ws in self.strategies.values()])

    def get_num_choices(self, unique=True) -> List[int]:
        """
        get the number of choices for every weight

        :param unique: False to consider weights that are used multiple times also multiple times in the indices
        :return: indices of the modules that should constitute the new architecture
                 a list of only ints if flat, otherwise a list of indices at every position
        """
        return [self.get_strategy_by_weight(n).get_requested_weight(n).num_choices()
                for n in self.ordered_names(unique=unique)]

    def get_num_weight_choices(self, name: str) -> int:
        """
        get the number of choices for a specific weight

        :param name: name of the weight
        """
        return self.get_strategy_by_weight(name).get_requested_weight(name).num_choices()

    def get_value_space(self, unique=True) -> ValueSpace:
        """
        get a value space based on the current strategy settings (order, num choices, masked, ...)

        :param unique: False to consider weights that are used multiple times also multiple times in the indices
        """
        num = [self.get_strategy_by_weight(n).get_requested_weight(n).get_choices()
               for n in self.ordered_names(unique=unique)]
        return ValueSpace(*[DiscreteValues(allowed_values=n) for n in num])

    def randomize_weights(self):
        """ randomizes all arc weights """
        for ws in self.strategies.values():
            ws.randomize_weights()

    def feedback(self, key: str, log_dict: dict, current_epoch: int, batch_idx: int):
        """
        feedback after each training forward pass
        this is currently not synchronized in distributed training

        :param key: train/val/test
        :param log_dict: contains loss, metric results, ...
        :param current_epoch:
        :param batch_idx:
        :return:
        """
        for ws in self.strategies.values():
            ws.feedback(key=key, log_dict=log_dict, current_epoch=current_epoch, batch_idx=batch_idx)


class StrategyManagerDefault:
    """
    context, execute code with a currently fixed strategy name

    with StrategyManagerDefault('my_strategy_name'):
        sm.make_weight(...)
        ...
    """

    def __init__(self, name: str, if_none=False):
        """
        :param name: name of the strategy to force-use during the with statement
        :param if_none: whether to force-use None (absence of a fixed strategy, use default)
        """
        self.sm = StrategyManager()
        self.name = name
        self._do = isinstance(name, str) or if_none
        self.prev_name = None

    def __enter__(self):
        if self._do:
            self.prev_name = self.sm.get_fixed_strategy_name()
            self.sm.set_fixed_strategy_name(self.name)

    def __exit__(self, *_, **__):
        if self._do:
            self.sm.set_fixed_strategy_name(self.prev_name)
