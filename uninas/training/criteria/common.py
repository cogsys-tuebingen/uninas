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
        return F.l1_loss(outputs, maybe_one_hot(outputs, targets), reduction=self.reduction)


@Register.criterion()
class L2Criterion(AbstractCriterion):
    """
    Mean squared difference between outputs and targets
    """

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(outputs, maybe_one_hot(outputs, targets), reduction=self.reduction)


@Register.criterion()
class HuberCriterion(AbstractCriterion):
    """
    Effectively the mean over all (output, target) pairs:
        L1 loss if L1(output, target) > delta
        L2 loss if L1(output, target) <= delta

    https://en.wikipedia.org/wiki/Huber_loss
    https://pytorch.org/docs/master/generated/torch.nn.HuberLoss.html
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('delta', default=1.0, type=float, help='use L1 when x < delta, otherwise L2'),
        ]

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # in torch 1.9 onwards:
        # return F.huber_loss(outputs, maybe_one_hot(outputs, targets))
        targets = maybe_one_hot(outputs, targets)
        l1 = F.l1_loss(outputs, targets, reduction='none').squeeze()
        larger = self.delta * (l1 - 0.5 * self.delta)
        smaller = 0.5 * F.mse_loss(outputs, targets, reduction='none').squeeze()
        cond = (l1 < self.delta)
        assert isinstance(cond, torch.Tensor)
        cond = cond.to(torch.float32)
        v = (smaller * cond) + (larger * (1 - cond))
        return self._reduce(v)


@Register.criterion()
class CrossEntropyCriterion(AbstractCriterion):
    """
    Cross-Entropy with optionally smoothed labels
    """

    def __init__(self, data_set: AbstractDataSet, **kwargs):
        super().__init__(data_set, **kwargs)
        assert isinstance(data_set, AbstractDataSet), "A data set is required to figure out the number of classes"
        self.num_classes = data_set.num_classes()
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
                return F.nll_loss(log_prob, targets, ignore_index=self.ignore_index, reduction=self.reduction)
            targets = torch.zeros_like(log_prob).scatter_(1, targets.unsqueeze(1), 1)
        if self.smoothing:
            targets = (1 - self.smoothing_epsilon) * targets + self.smoothing_epsilon / self.num_classes
        s = (-targets * log_prob)
        if self.ignore_index >= 0:
            s = s * (targets != self.ignore_index)
        return self._reduce(s).sum()
