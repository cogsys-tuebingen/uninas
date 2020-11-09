from uninas.utils.args import Argument
from uninas.register import Register
from uninas.methods.abstract import AbstractBiOptimizationMethod
from uninas.methods.strategies.manager import StrategyManager
from uninas.methods.strategies.gdas import GDASStrategy


@Register.method(search=True, single_path=True)
class GdasSearchMethod(AbstractBiOptimizationMethod):
    """
    Randomly sample 1 out of the available options, but requires shared topologies!
    Uses a gradient trick to compare which of the competing sampled paths were the most important
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('tau0', default=10, type=float, help='initial tau value for softmax annealing'),
            Argument('tauN', default=0.1, type=float, help='final tau value for softmax annealing'),
        ]

    def get_strategy(self):
        """ get strategy for architecture weights """
        tau0 = self._parsed_argument('tau0', self.hparams)
        return StrategyManager().add_strategy(GDASStrategy(self.max_epochs, tau0=tau0, use_mask=False))

    def _on_epoch_start(self) -> dict:
        log_dict = super()._on_epoch_start()
        tau0, tauN = self._parsed_arguments(['tau0', 'tauN'], self.hparams)
        ce, te = self.current_epoch, self.max_epochs
        self.strategy.tau = (tau0 - tauN) * (1 - ce/te) + tauN
        return self._add_to_dict(log_dict, dict(tau=self.strategy.tau))
