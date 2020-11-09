import torch
import torch.nn as nn
import torch.nn.functional as F
from uninas.register import Register
from uninas.methods.strategies.differentiable import DifferentiableStrategy


@Register.strategy(single_path=True)
class GDASStrategy(DifferentiableStrategy):
    """
    sample a single path,
    designed to be used with shared arc weights
    """

    def __init__(self, *args, tau0=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau0

    def forward(self, **__):
        # using log_sm instead of sm
        for n, w in self._weights.items():
            self._cached['sm'][n] = F.log_softmax(w / self.tau, dim=-1)

    def _combine_info(self, name: str) -> tuple:
        """ get a tuple or list of (idx, weight), instructing how to sum the path modules """
        w_sm = self.get_cached('sm', name)
        if self.use_mask:
            m = self.get_mask(name)
            w_sm = w_sm * m
            w_sm = w_sm / w_sm.sum()
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

    def _combine(self, name: str, x, modules: [nn.Module]) -> torch.Tensor:
        idx, w = self._combine_info(name)[0]
        return modules[idx](x) * w
