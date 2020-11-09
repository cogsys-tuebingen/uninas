from torch import nn
from uninas.methods.strategies.abstract import AbstractWeightStrategy
from uninas.utils.meta import Singleton


class StrategyManager(nn.Module, metaclass=Singleton):
    """
    Keeping track of (multiple) WeightStrategies
    """

    def __init__(self):
        super().__init__()
        self.strategies = {}            # {strategy name: strategy}
        self._requested = {}            # {weight name: strategy}
        self._ordered = []              # weight names in order of requests
        self._ordered_unique = []       # weight names in order of requests, ignoring duplicates

    def reset(self):
        self.strategies.clear()
        self._requested.clear()
        self._ordered.clear()
        self._ordered_unique.clear()

    def add_strategy(self, strategy: AbstractWeightStrategy):
        if strategy.name in self.strategies:
            raise KeyError('strategy name %s already in use' % strategy.name)
        self.strategies[strategy.name] = strategy
        return self

    def delete_strategy(self, name: str):
        strategy = self.strategies.get(name, None)
        if isinstance(strategy, AbstractWeightStrategy):
            for n in strategy.get_weight_names():
                self._requested.pop(n)
                self._ordered.remove(n)
                self._ordered_unique.remove(n)
            del self.strategies[name]
            del strategy
        return self

    def get_strategies(self) -> {str: AbstractWeightStrategy}:
        return self.strategies

    def get_strategies_list(self) -> list:
        return list(sorted(self.strategies.values(), key=lambda s: s.name))

    def make_weight(self, strategy_name: str, name: str,
                    choices: nn.ModuleList = None, num_choices: int = None) -> AbstractWeightStrategy:
        """
        register that a parameter of given name and num choices will be required during the search
        called by network components before the network is built
        """
        ws1 = self.strategies.get(strategy_name, None)
        ws2 = self._requested.get(name, None)
        num_choices = len(choices) if choices is not None else num_choices

        if ws1 is None:
            raise KeyError('strategy with name "%s" does not exist' % strategy_name)
        if num_choices < 1:
            raise ValueError('can not have a weight without option(s) to choose from')
        if ws2 not in [ws1, None]:
            raise ValueError('can not register the same weight name in different strategies')

        if name not in self._ordered:
            self._ordered_unique.append(name)
        self._ordered.append(name)
        self._requested[name] = ws1
        ws1.make_weight(name=name, choices=choices, num_choices=num_choices)
        return ws1

    def mask_index(self, idx: int, weight_name: str = None):
        """ prevent sampling a specific index, either for a specific weight or all weights """
        if isinstance(weight_name, str):
            strategies = [self._requested.get(weight_name)]
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
        :param fixed_arc: optional list of indices to fix the architecture, overwrites any fixed_arc in name_to_dict
        :param strategy_dict: {strategy name: strategy specific stuff}
        """
        if fixed_arc is not None:
            strategy_dict = {} if strategy_dict is None else strategy_dict
            # each strategy gets only the indices that belong to its weights
            for s in self.strategies.values():
                strategy_dict[s.name] = strategy_dict.get(s.name, {})
                strategy_dict[s.name]['fixed_arc'] = []
            for n, v in zip(self._ordered_unique, fixed_arc):
                strategy_dict[self._requested.get(n).name]['fixed_arc'].append(v)
        # if a strategy dict is given, execute only these strategies
        if strategy_dict is not None:
            for k, v in strategy_dict.items():
                self.strategies[k].forward(**v)
        else:
            for n, s in self.strategies.items():
                s.forward()

    def get_all_finalized_indices(self, unique=True) -> [[int]]:
        """
        :param unique: False to consider weights that are used multiple times also multiple times in the indices
        :return: indices of the modules that should constitute the new architecture
        """
        return [self._requested.get(n).get_finalized_indices(n) for n in self.ordered_names(unique=unique)]

    def ordered_names(self, unique=True) -> [str]:
        """
        name of each weight, by request order
        """
        lst = self._ordered_unique if unique else self._ordered
        return lst.copy()

    def ordered_num_choices(self, unique=True) -> [int]:
        """
        num choices per weight, by request order
        """
        lst = self._ordered_unique if unique else self._ordered
        return [self._requested.get(n).get_requested_weight(n).num_choices() for n in lst]

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

    def highest_value_per_weight(self) -> dict:
        """ {name: value} of the highest weight probability value """
        dct = {}
        for ws in self.strategies.values():
            dct.update(ws.highest_value_per_weight())
        return dct

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
