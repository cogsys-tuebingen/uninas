import numpy as np
import torch
import torch.nn as nn
from uninas.methods.abstract_strategy import AbstractWeightStrategy
from uninas.methods.abstract_method import AbstractOptimizationMethod
from uninas.methods.strategy_manager import StrategyManager
from uninas.register import Register


@Register.strategy(single_path=True, can_hpo=True)
class RandomChoiceStrategy(AbstractWeightStrategy):
    """
    This Strategy simply picks a random weight for each forward pass,
    it does not have any weights, therefore finalizing an architecture will be as the last sample.
    Should be used in conjunction with e.g. NSGA-II
    """

    def __init__(self, max_epochs: int, name='default', fixed_probabilities: [int] = None):
        super().__init__(max_epochs, name)
        self.fixed_probabilities = fixed_probabilities
        self._cached_idx = {}
        self._cached_max_idx = {}
        self._cached_idx_choices = {}
        if isinstance(self.fixed_probabilities, (list, tuple)) and len(self.fixed_probabilities) == 0:
            self.fixed_probabilities = None

    def build(self):
        for r in self._ordered_unique:
            self._cached_idx[r.name] = 0
            self._cached_max_idx[r.name] = r.num_choices()
            self._cached_idx_choices[r.name] = np.arange(0, r.num_choices())
        # probabilities
        if isinstance(self.fixed_probabilities, list):
            sizes = [len(v) for v in self._cached_idx_choices.values()]
            mi, ma, fp = min(sizes), max(sizes), len(self.fixed_probabilities)
            assert mi == ma == fp,\
                "For fixed probabilities, all arc weights must have the same length! (%d/%d/%d)" % (mi, ma, fp)

    def randomize_weights(self):
        """ randomizes all arc weights """
        for n in self.get_weight_names():
            self._cached_idx[n] = np.random.choice(self._cached_idx_choices[n], size=1, p=self.fixed_probabilities)[0]

    def forward(self, fixed_arc=None, **__):
        """
        set the architecture for future steps
        :param fixed_arc: desired architecture, random if None
        """
        if fixed_arc is None:
            self.randomize_weights()
        else:
            for r, w in zip(self._ordered_unique, fixed_arc):
                self._cached_idx[r.name] = w

    def get_weight_sm(self, name: str) -> torch.Tensor:
        """ softmax over the specified weight """
        w = torch.zeros(size=[self._cached_max_idx[name]])
        idx = self._cached_idx[name]
        if idx >= 0:
            w[idx] = 1
        return w

    def get_finalized_indices(self, name: str) -> [int]:
        """ return indices of the modules that should constitute the new architecture, for this specific weight """
        return [self._cached_idx[name]]

    def _combine_info(self, name: str) -> tuple:
        """ get a tuple or list of (idx, weight), instructing how to sum the path modules """
        return (self._cached_idx[name], 1.0),

    def combine(self, name: str, x, modules: [nn.Module]) -> torch.Tensor:
        """
        combine multiple outputs into one, depending on arc weights

        :param name: name of the SearchModule object
        :param x: input (e.g. torch.Tensor)
        :param modules: torch.nn.Modules, may be None if module_results are available
        :return: combination of module results
        """
        return modules[self._cached_idx[name]](x)

    def _mask_index(self, idx: int, weight_name: str):
        x = self._cached_idx_choices[weight_name]
        self._cached_idx_choices[weight_name] = x[x != idx]


@Register.method(search=True, single_path=True, can_hpo=True)
class UniformRandomMethod(AbstractOptimizationMethod):
    """
    Randomly sample 1 out of the available options

    Should be used in conjunction with e.g. NSGA-II after the super-network training
    """

    def setup_strategy(self) -> StrategyManager:
        """ set up the strategy for architecture weights """
        return StrategyManager().add_strategy(RandomChoiceStrategy(self.max_epochs))
