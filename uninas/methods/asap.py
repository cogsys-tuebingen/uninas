from uninas.utils.args import Argument
from uninas.register import Register
from uninas.methods.abstract_method import AbstractBiOptimizationMethod
from uninas.methods.strategy_manager import StrategyManager
from uninas.methods.darts import DifferentiableStrategy


@Register.strategy()
class AsapStrategy(DifferentiableStrategy):

    def __init__(self, *args, tau=1.6, tau_grace=1.0, beta=0.95, mask_below=0.4, mask_by_numel=True, **kwargs):
        super().__init__(*args, tau=tau, **kwargs)
        self.tau0 = tau
        self.tau_grace = tau_grace
        self.beta = beta
        self.mask_below = mask_below
        self.mask_by_numel = mask_by_numel

    def on_epoch_start(self, current_epoch: int):
        self.tau = self.tau0 * self.beta ** self.current_epoch
        self.mask_all_weights_below(self.mask_below, div_by_numel=self.mask_by_numel)


@Register.method(search=True)
class AsapSearchMethod(AbstractBiOptimizationMethod):
    """
    Executes all choices, learns how to weights them in a weighted sum,
    anneals the softmax temperature to enforce convergence and prunes the options that are weighted below a threshold

    ASAP: Architecture Search, Anneal and Prune
    https://arxiv.org/abs/1904.04123
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
        tau_0, tau_grace, beta = self._parsed_arguments(['tau_0', 'tau_grace', 'beta'], self.hparams)
        return StrategyManager().add_strategy(AsapStrategy(self.max_epochs, tau=tau_0, tau_grace=tau_grace,
                                                           beta=beta, use_mask=True,
                                                           mask_below=0.4, mask_by_numel=True))

    def _on_epoch_start(self) -> dict:
        log_dict = super()._on_epoch_start()
        for strategy in StrategyManager().get_strategies_list():
            assert isinstance(strategy, AsapStrategy)
            if strategy.tau < strategy.tau_grace:
                self.set_loader_multiples((1, 1))
            else:
                self.set_loader_multiples((1, 0))
        return log_dict
