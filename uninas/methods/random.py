from uninas.register import Register
from uninas.methods.abstract import AbstractOptimizationMethod
from uninas.methods.strategies.manager import StrategyManager
from uninas.methods.strategies.random import RandomChoiceStrategy, FairRandomChoiceStrategy
from uninas.utils.args import Namespace


@Register.method(search=True, single_path=True, can_hpo=True)
class UniformRandomMethod(AbstractOptimizationMethod):
    """
    Randomly sample 1 out of the available options
    """

    def setup_strategy(self) -> StrategyManager:
        """ set up the strategy for architecture weights """
        return StrategyManager().add_strategy(RandomChoiceStrategy(self.max_epochs))


@Register.method(search=True, single_path=True, can_hpo=True)
class StrictlyFairRandomMethod(AbstractOptimizationMethod):
    """
    Randomly sample 1 out of the available options in a strictly fair way,
    so that within n steps, each of the n available options was picked exactly once
    """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        self.steps_for_update = self.strategy_manager.max_num_choices()
        self.steps_last_update = 0

    def setup_strategy(self) -> StrategyManager:
        """ set up the strategy for architecture weights """
        return StrategyManager().add_strategy(FairRandomChoiceStrategy(self.max_epochs, assert_same_length=True))

    def optimizer_step(self, *args, **kwargs):
        """ only have a parameter update every n steps, when every path has received exactly one gradient """
        self.steps_last_update = (self.steps_last_update + 1) % self.steps_for_update
        if self.steps_last_update == 0:
            super().optimizer_step(*args, **kwargs)
