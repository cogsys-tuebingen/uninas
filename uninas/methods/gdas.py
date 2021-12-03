import torch
import torch.nn.functional as F
from uninas.methods.abstract_method import AbstractBiOptimizationMethod
from uninas.methods.strategy_manager import StrategyManager
from uninas.methods.darts import DifferentiableStrategy
from uninas.utils.args import Argument
from uninas.register import Register


@Register.strategy(single_path=True)
class GDASStrategy(DifferentiableStrategy):
    """
    sample a single path,
    designed to be used with shared arc weights
    """

    def __init__(self, *args, tau0=10.0, tau_n=0.1, use_mask=True, **kwargs):
        super().__init__(*args, use_mask=use_mask, tau=tau0, **kwargs)
        self.tau0 = tau0
        self.tau_n = tau_n

    def forward(self, **__):
        # using log_sm instead of sm
        for n, w in self.weights.items():
            self._cached['sm'][n] = F.log_softmax(w / self.tau, dim=-1)

    def _combine_info(self, name: str) -> tuple:
        """ get a tuple or list of (idx, weight), instructing how to sum the path modules """
        w_sm = self.get_weight_sm(name)
        while True:
            gumbels = -torch.empty_like(w_sm).exponential_().log()
            logits = (w_sm + gumbels) / self.tau
            probs = F.softmax(logits, dim=-1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = F.one_hot(index, num_classes=probs.shape[0])[0].float()
            weights = one_h - probs.detach() + probs
            if torch.isinf(gumbels).any() or torch.isinf(probs).any() or torch.isnan(probs).any():
                continue
            break
        return (index, weights[index]),

    def on_epoch_start(self, current_epoch: int):
        self.tau = (self.tau0 - self.tau_n) * (1 - current_epoch / self.max_epochs) + self.tau_n


@Register.method(search=True, single_path=True)
class GdasSearchMethod(AbstractBiOptimizationMethod):
    """
    Randomly sample 1 out of the available options, but requires shared topologies!
    Uses a gradient trick to compare which of the competing sampled paths were the most important

    Searching for A Robust Neural Architecture in Four GPU Hours
    https://arxiv.org/abs/1910.04465
    https://github.com/D-X-Y/AutoDL-Projects
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('tau0', default=10, type=float, help='initial tau value for softmax annealing'),
            Argument('tauN', default=0.1, type=float, help='final tau value for softmax annealing'),
        ]

    def setup_strategy(self) -> StrategyManager:
        """ set up the strategy for architecture weights """
        tau0, tau_n = self._parsed_arguments(['tau0', 'tauN'], self.hparams)
        return StrategyManager().add_strategy(GDASStrategy(self.max_epochs, tau0=tau0, tau_n=tau_n))
