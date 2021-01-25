"""
random strategies to (de)activate paths

turns out numpy.random is deterministic while torch.random is not,
probably since torch also uses its random variables in other threads, e.g. to load data
"""

import numpy as np
import torch
import torch.nn as nn
from uninas.register import Register
from uninas.methods.strategies.abstract import AbstractWeightStrategy


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

    def highest_value_per_weight(self) -> dict:
        """ {name: value} of the highest weight probability value """
        return {n: 0.0 for n in self.get_weight_names()}

    def get_finalized_indices(self, name: str) -> [int]:
        """ return indices of the modules that should constitute the new architecture, for this specific weight """
        return [self._cached_idx[name]]

    def _combine_info(self, name: str) -> tuple:
        """ get a tuple or list of (idx, weight), instructing how to sum the path modules """
        return (self._cached_idx[name], 1.0),

    def _combine(self, name: str, x, modules: [nn.Module]) -> torch.Tensor:
        return modules[self._cached_idx[name]](x)

    def _mask_index(self, idx: int, weight_name: str):
        x = self._cached_idx_choices[weight_name]
        self._cached_idx_choices[weight_name] = x[x != idx]


@Register.strategy(single_path=True, can_hpo=True)
class FairRandomChoiceStrategy(RandomChoiceStrategy):
    """
    This Strategy simply picks a random weight for each forward pass in a strictly fair way,
    so that every num_choices steps, each path is chosen exactly once
    it does not have any weights, therefore finalizing an architecture will be as the last sample.
    Should be used in conjunction with e.g. NSGA-II
    """

    def __init__(self, max_epochs: int, name='default', assert_same_length=True):
        super().__init__(max_epochs, name)
        self._cached_next_idxs = {}
        self.assert_same_length = assert_same_length

    def build(self):
        super().build()
        for r in self._ordered_unique:
            self._cached_next_idxs[r.name] = []
            self._cached_idx[r.name] = 0
        if self.assert_same_length:
            sizes = [len(v) for v in self._cached_idx_choices.values()]
            assert min(sizes) == max(sizes), "For strict fairness, all arc weights must have the same length!"

    def randomize_weights(self):
        """ randomizes all arc weights in a strictly fair way """
        for n, lst in self._cached_next_idxs.items():
            if len(lst) == 0:
                self._cached_next_idxs[n] = np.random.permutation(self._cached_idx_choices[n]).tolist()
            self._cached_idx[n] = self._cached_next_idxs[n].pop()
