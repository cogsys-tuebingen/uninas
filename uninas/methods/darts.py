from uninas.register import Register
from uninas.methods.abstract import AbstractBiOptimizationMethod
from uninas.methods.strategies.manager import StrategyManager
from uninas.methods.strategies.differentiable import DifferentiableStrategy


@Register.method(search=True)
class DartsSearchMethod(AbstractBiOptimizationMethod):
    """
    Executes all choices, learns how to weights them in a weighted sum
    """

    def get_strategy(self):
        """ get strategy for architecture weights """
        return StrategyManager().add_strategy(DifferentiableStrategy(self.max_epochs, use_mask=False))
