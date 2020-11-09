from uninas.utils.args import Argument
from uninas.register import Register
from uninas.methods.abstract import AbstractOptimizationMethod
from uninas.methods.strategies.manager import StrategyManager
from uninas.methods.strategies.mdenas import MdlStrategy


@Register.method(search=True, single_path=True)
class MdlSearchMethod(AbstractOptimizationMethod):
    """
    Randomly sample 1 out of the available options,
    use validation feedback to rank them
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('key', default='val/accuracy/1', type=str, help='key to optimize'),
            Argument('alpha', default=0.01, type=float, help='update rate for probability distributions'),
            Argument('grace_epochs', default=0, type=int, help='grace epochs before probability updates'),
        ]

    def get_strategy(self):
        """ get strategy for architecture weights """
        key, alpha, grace_epochs = self._parsed_arguments(['key', 'alpha', 'grace_epochs'], self.hparams)
        return StrategyManager().add_strategy(
            MdlStrategy(self.max_epochs, key=key, alpha=alpha, grace_epochs=grace_epochs))
