from typing import Union
import torch
import torch.nn as nn
from uninas.data.abstract import AbstractDataSet
from uninas.utils.args import ArgsInterface, Namespace, Argument


class AbstractCriterion(nn.Module, ArgsInterface):
    def __init__(self, data_set: Union[AbstractDataSet, None], **kwargs):
        nn.Module.__init__(self)
        ArgsInterface.__init__(self)
        defaults = self.parsed_argument_defaults()
        defaults.update(kwargs)
        for k, v in defaults.items():
            self.__setattr__(k, v)

    @classmethod
    def from_args(cls, data_set: Union[AbstractDataSet, None], args: Namespace, index: int = None) -> 'AbstractCriterion':
        all_parsed = cls._all_parsed_arguments(args, index=index)
        return cls(data_set, **all_parsed)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('reduction', default="mean", type=str, choices=["mean", "sum", "none"], help='how to reduce the error'),
        ]

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        """ reduce x according to the set 'reduction' """
        if self.reduction == "mean":
            return x.mean(0)
        if self.reduction == "sum":
            return x.sum(0)
        return x


class MultiCriterion(nn.Module):
    """
    wraps a normal criterion to be applicable to multiple model outputs, weights their respective loss values
    """

    def __init__(self, criterion: AbstractCriterion, weights: [float]):
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

    @classmethod
    def from_args(cls, head_weights: [float], criterion_cls: AbstractCriterion.__class__,
                  data_set: AbstractDataSet, args: Namespace) -> 'MultiCriterion':
        criterion = criterion_cls.from_args(data_set, args, index=None)
        return cls(criterion, head_weights)
