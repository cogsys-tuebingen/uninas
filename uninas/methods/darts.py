from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from uninas.methods.abstract_method import AbstractBiOptimizationMethod
from uninas.methods.strategy_manager import StrategyManager
from uninas.methods.abstract_strategy import AbstractWeightStrategy
from collections.abc import Iterable
from uninas.register import Register


@Register.strategy()
class DifferentiableStrategy(AbstractWeightStrategy):
    """
    weighted sum of all choices
    computes and caches the softmax over all weights with same length at once
    """

    def __init__(self, max_epochs: int, name='default', mul=0.001, tau=1.0, use_mask=False):
        """
        :param max_epochs: max number of epochs
        :param name: name of this particular WeightStrategy, can access them via name from the class alone
        :param mul: multiplicative factor for normally distributed random architecture weights
        :param tau: annealing value for the softmax function
        :param use_mask: can mask weights to ignore their respective paths
        """
        super().__init__(max_epochs, name)
        self._name_to_weight = {}       # {name: (num_choices, row idx in matrix)}

        self.mul = mul
        self.tau = tau
        self.use_mask = use_mask
        self.weights = nn.ParameterDict()
        self.masks = nn.ParameterDict()
        self._cached = {'sm': {}}
        self._fixed = {}

    def get_log_dict(self) -> {str: float}:
        """
        :return: dict of values that are interesting to log
        """
        dct = super().get_log_dict()
        if self.use_mask:
            remaining, total = self.get_masks_remaining_total()
            dct.update({
                'masks/remaining/relative': remaining / total,
                'masks/remaining/total': remaining,
                'masks/total': total,
            })
        dct['tau'] = self.tau
        dct.update({"max_value/%s" % n: self.get_weight_sm(n).max().item() for n, _ in self.named_parameters_single()})
        return dct

    def named_parameters_single(self, prefix='', recurse=True) -> Iterable:
        # returns each arc weight on its own, not grouped, nicer inspection
        for name in self._name_to_weight.keys():
            yield prefix+name, self.get_weight(name)

    def build(self):
        """
        actually generate all the requested weights, group weights with same num params for efficiency
        called once after the network is built
        """
        # create weights, ...
        for k, v in self._requested_sizes().items():
            name = str(k)
            # weights and masks
            self.weights[name] = nn.Parameter(self.mul * torch.randn([len(v), k]), requires_grad=True)
            self.masks[name] = nn.Parameter(torch.ones_like(self.weights[name]), requires_grad=False)
            # link to weight etc by name
            for i, r in enumerate(v):
                self._name_to_weight[r.name] = (name, i)

    def forward(self, fixed_arc=None, **__):
        """
        called once before every network forward pass
        pre-compute values of interest, e.g. softmax over weights
        """
        if fixed_arc is None:
            self._fixed.clear()
            for n, w in self.weights.items():
                self._cached['sm'][n] = self._compute_softmax(n)
        else:
            for r, w in zip(self._ordered_unique, fixed_arc):
                self._fixed[r.name] = w

    def _get_by_name(self, name: str, dct: Union[dict, nn.ParameterDict]):
        k, n = self._name_to_weight.get(name)
        return dct[k][n]

    def randomize_weights(self):
        """ randomizes all arc weights """
        for w in self.weights.values():
            w.data.zero_().add_(self.mul * torch.randn_like(w).to(w.device))

    def get_weight(self, name: str) -> torch.Tensor:
        """ get a single weight tensor by name """
        return self._get_by_name(name, self.weights)

    def get_cached(self, key: str, name: str) -> torch.Tensor:
        """ get a single cached tensor by key and name """
        # may fail e.g. after loading weights, since then the cache is not initialized
        try:
            return self._get_by_name(name, self._cached[key])
        except KeyError:
            self.forward()
            return self._get_by_name(name, self._cached[key])

    def get_weight_sm(self, name: str) -> torch.Tensor:
        """ get the cached weight softmax for a single weight """
        return self.get_cached('sm', name)

    def _compute_softmax(self, name: str, masked: bool = None) -> torch.Tensor:
        """
        compute the (masked) softmax for a grouped weight

        :param name: name of the grouped weight
        :param masked: apply mask to weights. use self.use_mask if None
        :return: softmax over (masked) weights
        """
        w_sm = F.softmax(self.weights[name] / self.tau, dim=-1)
        if masked or (self.use_mask and masked is None):
            w_sm = w_sm * self.masks[name]
            w_sm = w_sm / w_sm.sum(dim=-1).view(-1, 1)
        return w_sm

    def get_finalized_indices(self, name: str) -> [int]:
        """ return indices of the modules that should constitute the new architecture, for this specific weight """
        return [torch.argmax(self.get_weight_sm(name), dim=-1).item()]

    def _mask_index(self, idx: int, weight_name: str):
        x = self.get_mask(weight_name)
        x[idx].zero_()
        self.use_mask = True

    def get_mask(self, name) -> torch.Tensor:
        return self._get_by_name(name, self.masks)

    def get_masks_remaining_total(self) -> (int, int):
        """ count paths that are not masked out and total number of paths """
        remaining, total = 0, 0
        for m in self.masks.values():
            remaining += m.sum().item()
            total += m.numel()
        return int(remaining), total

    def mask_all_weights_below(self, p=0.05, div_by_numel=False):
        """ masks weights below sm probability p """
        self.use_mask = True
        for name in self._name_to_weight.keys():
            w_sm, m = self.get_weight_sm(name), self.get_mask(name)
            if div_by_numel:
                p /= w_sm.numel()
            pos = w_sm < p
            m.data[pos] = 0

    def on_epoch_start(self, current_epoch: int):
        """ whenever the method starts a new epoch """
        # simply make sure that there is a cached softmax on the correct device
        if len(self._cached['sm']) == 0:
            self.forward()

    def on_epoch_end(self, current_epoch: int) -> bool:
        """
        whenever the method ends an epoch
        signal early stopping when returning True
        """
        for name in self._name_to_weight.keys():
            if self.get_mask(name).sum() > 1:
                return False
        return True

    def _combine_info(self, name: str) -> list:
        """ get a tuple or list of (idx, weight), instructing how to sum the path modules """
        if self._fixed.get(name, False):
            return self._fixed.get(name)
        w_sm = self.get_weight_sm(name)
        return [(i, w) for i, w in enumerate(w_sm) if w > 0.0]

    def combine(self, name: str, x, modules: [nn.Module]) -> torch.Tensor:
        """
        combine multiple outputs into one, depending on arc weights

        :param name: name of the SearchModule object
        :param x: input (e.g. torch.Tensor)
        :param modules: torch.nn.Modules, may be None if module_results are available
        :return: combination of module results
        """
        if self._fixed.get(name, False):
            return modules[self._fixed.get(name)](x)
        combine_info = self._combine_info(name)
        if len(combine_info) == 1:
            idx, w = combine_info[0]
            return modules[idx](x) * w
        return sum([modules[i](x) * w for (i, w) in combine_info])


@Register.method(search=True)
class DartsSearchMethod(AbstractBiOptimizationMethod):
    """
    Executes all choices, learns how to weights them in a weighted sum

    DARTS: Differentiable Architecture Search
    https://arxiv.org/abs/1806.09055
    https://github.com/quark0/darts
    """

    def setup_strategy(self) -> StrategyManager:
        """ set up the strategy for architecture weights """
        return StrategyManager().add_strategy(DifferentiableStrategy(self.max_epochs, use_mask=False))
