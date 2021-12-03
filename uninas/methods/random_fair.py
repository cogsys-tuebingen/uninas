from typing import Callable, Optional
import numpy as np
from uninas.methods.abstract_method import AbstractOptimizationMethod
from uninas.methods.random import RandomChoiceStrategy
from uninas.methods.strategy_manager import StrategyManager
from uninas.utils.args import Namespace
from uninas.register import Register


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


@Register.method(search=True, single_path=True, can_hpo=True)
class StrictlyFairRandomMethod(AbstractOptimizationMethod):
    """
    Randomly sample 1 out of the available options in a strictly fair way,
    so that within n steps, each of the n available options was picked exactly once

    Should be used in conjunction with e.g. NSGA-II after the super-network training

    FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search
    https://arxiv.org/abs/1907.01845
    """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        self.steps_for_update = self.strategy_manager.max_num_choices()
        self.steps_last_update = 0

    def setup_strategy(self) -> StrategyManager:
        """ set up the strategy for architecture weights """
        return StrategyManager().add_strategy(FairRandomChoiceStrategy(self.max_epochs, assert_same_length=True))

    def optimizer_step(self, *args, optimizer_closure: Optional[Callable] = None, **kwargs):
        """ only have a parameter update every n steps, when every path has received exactly one gradient """
        self.steps_last_update = (self.steps_last_update + 1) % self.steps_for_update
        if self.steps_last_update == 0:
            super().optimizer_step(*args, optimizer_closure=optimizer_closure, **kwargs)
        else:
            # execute the closure in any case, which probably calls loss.backward()
            if isinstance(optimizer_closure, Callable):
                optimizer_closure()
