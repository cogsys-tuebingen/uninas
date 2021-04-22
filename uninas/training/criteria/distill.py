import torch
import torch.nn.functional as F
from uninas.training.criteria.abstract import AbstractCriterion
from uninas.register import Register


@Register.criterion(distill=True)
class DistillL2Criterion(AbstractCriterion):
    """
    the model outputs also contain the guiding signal,
    the true label is ignored (shape will not match anyway)
    """
    def forward(self, outputs: [(torch.Tensor, torch.Tensor)], targets: torch.Tensor) -> torch.Tensor:
        return sum([F.mse_loss(o1, o2) for o1, o2 in outputs])
