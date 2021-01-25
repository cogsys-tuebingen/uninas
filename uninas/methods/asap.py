from uninas.utils.args import Argument
from uninas.register import Register
from uninas.methods.abstract import AbstractBiOptimizationMethod
from uninas.methods.strategies.manager import StrategyManager
from uninas.methods.strategies.differentiable import DifferentiableStrategy


@Register.method(search=True)
class AsapSearchMethod(AbstractBiOptimizationMethod):
    """
    Executes all choices, learns how to weights them in a weighted sum,
    anneals the softmax temperature to enforce convergence and prunes the options that are weighted below a threshold
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('tau_0', default=1.6, type=float, help='initial tau value for the softmax temperature'),
            Argument('tau_grace', default=1.0, type=float, help='no arc training/pruning until tau is smaller'),
            Argument('beta', default=0.95, type=float, help='beta value to anneal tau0'),
        ]

    def setup_strategy(self) -> StrategyManager:
        """ set up the strategy for architecture weights """
        tau_0 = self._parsed_argument('tau_0', self.hparams)
        return StrategyManager().add_strategy(DifferentiableStrategy(self.max_epochs, tau=tau_0, use_mask=True))

    def _on_epoch_start(self) -> dict:
        log_dict = super()._on_epoch_start()
        tau_0, tau_grace, beta = self._parsed_arguments(['tau_0', 'tau_grace', 'beta'], self.hparams)
        for strategy in StrategyManager().get_strategies_list():
            strategy.tau = tau_0 * beta ** self.current_epoch
            log_dict = self._add_to_dict(log_dict, dict(tau=strategy.tau))
            self.update_architecture_weights = strategy.tau < tau_grace
            if self.update_architecture_weights:
                strategy.mask_all_weights_below(0.4, div_by_numel=True)
                log_dict.update(strategy.get_masks_log_dict(prefix='asap/masks'))
                self.set_loader_multiples((1, 1))
            else:
                self.set_loader_multiples((1, 0))
            return log_dict
