import torch
import torch.nn.functional as F
from uninas.training.criteria.abstract import AbstractCriterion
from uninas.data.abstract import AbstractDataSet
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


def maybe_one_hot(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ cast the targets into one-hot if the shapes mismatch, the last dimension must be the class """
    if outputs.shape == targets.shape:
        return targets.to(outputs.dtype)
    return F.one_hot(targets, outputs.shape[-1]).to(outputs.dtype)


@Register.criterion()
class L1Criterion(AbstractCriterion):
    """
    Mean absolute difference between outputs and targets
    """

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(outputs, maybe_one_hot(outputs, targets))


@Register.criterion()
class RelativeL1Criterion(AbstractCriterion):
    """
    Compute the mean (L1(output, target) / target)
    """

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(outputs, maybe_one_hot(outputs, targets), reduction="none")
        return (l1 / targets).mean()


@Register.criterion()
class L2Criterion(AbstractCriterion):
    """
    Mean squared difference between outputs and targets
    """

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(outputs, maybe_one_hot(outputs, targets))


@Register.criterion()
class Huber1Criterion(AbstractCriterion):
    """
    Mean over all (output, target) pairs:
        L1 loss if L1(output, target) < delta
        L2 loss if L1(output, target) >= delta
    """
    _delta = 1.0

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # in torch 1.9 onwards:
        # return F.huber_loss(outputs, maybe_one_hot(outputs, targets))
        targets = maybe_one_hot(outputs, targets)
        l1 = F.l1_loss(outputs, targets, reduction='none').squeeze()
        l2 = F.mse_loss(outputs, targets, reduction='none').squeeze()
        cond = (l1 < self._delta).to(torch.float32)
        v = (l1 * cond) + (l2 * (1 - cond))
        return v.mean()


@Register.criterion()
class CrossEntropyCriterion(AbstractCriterion):
    """
    Cross-Entropy with optionally smoothed labels
    """

    def __init__(self, args: Namespace, data_set: AbstractDataSet):
        super().__init__(args, data_set)
        self.num_classes = data_set.num_classes()
        self.smoothing_epsilon = self._parsed_argument('smoothing_epsilon', args)
        self.ignore_index = self._parsed_argument('ignore_index', args)
        self.smoothing = self.smoothing_epsilon > 0

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('smoothing_epsilon', default=0.0, type=float, help='label smoothing, <=0 to disable'),
            Argument('ignore_index', default=-100, type=int, help='ignore classes with index')
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
                return F.nll_loss(log_prob, targets, ignore_index=self.ignore_index)
            targets = torch.zeros_like(log_prob).scatter_(1, targets.unsqueeze(1), 1)
        if self.smoothing:
            targets = (1 - self.smoothing_epsilon) * targets + self.smoothing_epsilon / self.num_classes
        s = (-targets * log_prob)
        if self.ignore_index >= 0:
            s = s * (targets != self.ignore_index)
        return s.mean(0).sum()
