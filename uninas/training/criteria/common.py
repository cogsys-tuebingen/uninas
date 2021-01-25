import torch
import torch.nn.functional as F
from uninas.training.criteria.abstract import AbstractCriterion
from uninas.data.abstract import AbstractDataSet
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


def maybe_one_hot(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ cast the targets into one-hot if the shapes mismatch, the last dimension must be the class """
    if outputs.shape == targets.shape:
        return targets
    return F.one_hot(targets, outputs.shape[-1]).to(outputs.dtype)


@Register.criterion()
class L1Criterion(AbstractCriterion):
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(outputs, maybe_one_hot(outputs, targets))


@Register.criterion()
class L2Criterion(AbstractCriterion):
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(outputs, maybe_one_hot(outputs, targets))


@Register.criterion()
class CrossEntropyCriterion(AbstractCriterion):
    """
    Cross-Entropy with optionally smoothed labels
    """

    def __init__(self, args: Namespace, data_set: AbstractDataSet):
        super().__init__(args, data_set)
        self.num_classes = data_set.num_classes()
        self.smoothing_epsilon = self._parsed_argument('smoothing_epsilon', args)
        self.smoothing = self.smoothing_epsilon > 0

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('smoothing_epsilon', default=0.0, type=float, help='label smoothing, <=0 to disable')
        ]

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param outputs:
        :param targets: tensor, either size [batch] with class indices or [batch, class weights]
        :return: averaged loss
        """
        log_prob = F.log_softmax(outputs, dim=1)
        if len(targets.shape) == 1:
            if not self.smoothing:
                return F.nll_loss(log_prob, targets)
            targets = torch.zeros_like(log_prob).scatter_(1, targets.unsqueeze(1), 1)
        if self.smoothing:
            targets = (1 - self.smoothing_epsilon) * targets + self.smoothing_epsilon / self.num_classes
        return (-targets * log_prob).mean(0).sum()
