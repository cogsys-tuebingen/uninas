import torch
import torch.nn as nn

from uninas.data.abstract import AbstractDataSet
from uninas.utils.args import ArgsInterface, Namespace


class MultiCriterion(nn.Module):
    """
    wraps a normal criterion to be applicable to multiple model outputs, weights their respective loss values
    """

    def __init__(self, criterion, weights: [float]):
        super().__init__()
        self.criterion = criterion
        self.weights = weights

    def __str__(self):
        return '%s(%s, weighted: %s)' % (self.__class__.__name__, self.criterion.__class__.__name__, str(self.weights))

    def forward(self, outputs: [torch.Tensor], target: torch.Tensor):
        losses = []
        for output, weight in zip(outputs, self.weights):
            if (output is None) or (weight is None) or (weight == 0.0):
                continue
            losses.append(self.criterion(output, target) * weight)
        return sum(losses)


class MetaMultiCriterion(type):
    """ meta class that automatically wraps the criterion in a MultiCriterion """

    def __call__(cls, weights, *args, **kwargs):
        criterion = super(MetaMultiCriterion, cls).__call__(*args, **kwargs)
        return MultiCriterion(criterion, weights)


class AbstractCriterion(nn.Module, ArgsInterface, metaclass=MetaMultiCriterion):
    def __init__(self, args: Namespace, data_set: AbstractDataSet):
        nn.Module.__init__(self)
        ArgsInterface.__init__(self)
