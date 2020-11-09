from collections.abc import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from uninas.register import Register
from uninas.methods.strategies.abstract import AbstractWeightStrategy


@Register.strategy()
class DifferentiableStrategy(AbstractWeightStrategy):
    """
    weighted sum of all choices
    computes and caches the softmax over all weights with same length at once
    """

    def __init__(self, max_epochs: int, name='default', mul=1e-3, tau=1.0, use_mask=False):
        """
        :param max_epochs: max number of epochs
        :param name: name of this particular WeightHelper, can access them via name from the class alone
        :param mul: multiplicative factor for normally distributed random architecture weights
        :param tau: annealing value for the softmax function
        :param use_mask: can mask weights to ignore their respective paths
        """
        super().__init__(max_epochs, name)
        self._name_to_weight = {}       # {name: (num_choices, row idx in matrix)}

        self.mul = mul
        self.tau = tau
        self.use_mask = use_mask
        self._weights = nn.ParameterDict()
        self._masks = nn.ParameterDict()
        self._cached = {'sm': {}}
        self._fixed = {}

    def parameters(self, recurse=True) -> Iterable:
        return list(self._weights.values()) + list(self._masks.values())

    def named_parameters(self, prefix='', recurse=True) -> Iterable:
        for k, v in self._weights.items():
            yield prefix+k, v
        for k, v in self._masks.items():
            yield prefix+k, v

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
            self._weights[name] = nn.Parameter(self.mul * torch.randn([len(v), k]), requires_grad=True)
            self._masks[name] = nn.Parameter(torch.ones_like(self._weights[name]), requires_grad=False)
            # link to weight etc by name
            for i, r in enumerate(v):
                self._name_to_weight[r.name] = (name, i)
            # cached values
            for c in self._cached.keys():
                self._cached[c][name] = torch.zeros_like(self._weights[name], device=self._weights[name].device)
        self.forward()

    def forward(self, fixed_arc=None, **__):
        """
        called once before every network forward pass
        pre-compute values of interest, e.g. softmax over weights
        """
        if fixed_arc is None:
            self._fixed.clear()
            for n, w in self._weights.items():
                self._cached['sm'][n] = F.softmax(w / self.tau, dim=-1)
        else:
            for r, w in zip(self._ordered_unique, fixed_arc):
                self._fixed[r.name] = w

    def _get_by_name(self, name, dct):
        k, n = self._name_to_weight.get(name)
        return dct[k][n]

    def randomize_weights(self):
        """ randomizes all arc weights """
        for w in self._weights.values():
            w.data.zero_().add_(self.mul * torch.randn_like(w).to(w.device))

    def get_weight(self, name) -> torch.Tensor:
        return self._get_by_name(name, self._weights)

    def get_cached(self, key, name) -> torch.Tensor:
        return self._get_by_name(name, self._cached[key])

    def get_weight_sm(self, name: str) -> torch.Tensor:
        """ softmax over weights """
        return self.get_cached('sm', name)

    def highest_value_per_weight(self) -> dict:
        """ {name: value} of the highest weight probability value """
        return {n: self.get_weight_sm(n).max().item() for n, _ in self.named_parameters_single()}

    def get_finalized_indices(self, name: str) -> [int]:
        """ return indices of the modules that should constitute the new architecture, for this specific weight """
        return [torch.argmax(self.get_weight(name), dim=-1).item()]

    def _mask_index(self, idx: int, weight_name: str):
        x = self.get_mask(weight_name)
        x[idx].zero_()
        self.use_mask = True

    def get_mask(self, name) -> torch.Tensor:
        return self._get_by_name(name, self._masks)

    def get_masks_log_dict(self, prefix='method') -> dict:
        remaining, total = self.get_masks_remaining_total()
        return {
            prefix+'/relative': remaining / total,
            prefix+'/remaining': remaining,
            prefix+'/total': total,
        }

    def get_masks_remaining_total(self) -> (int, int):
        """ count paths that are not masked out and total number of paths """
        remaining, total = 0, 0
        for m in self._masks.values():
            remaining += m.sum().item()
            total += m.numel()
        return int(remaining), total

    def mask_all_weights_below(self, p=0.05, div_by_numel=False):
        """ masks weights below sm probability p """
        for name in self._name_to_weight.keys():
            w_sm, m = self.get_weight_sm_masked(name)
            if div_by_numel:
                p /= w_sm.numel()
            pos = w_sm < p
            m.data[pos] = 0

    def get_weight_sm_masked(self, name) -> (torch.Tensor, torch.Tensor):
        """ masked softmax over weights and mask """
        sm, m = self.get_weight_sm(name), self.get_mask(name)
        sm = sm * m
        sm = sm / sm.sum()
        return sm, m

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
        if self.use_mask:
            w_sm, mask = self.get_weight_sm_masked(name)
            return [(i, w) for i, (w, m) in enumerate(zip(w_sm, mask)) if m]
        w_sm = self.get_weight_sm(name)
        return [(i, w) for i, w in enumerate(w_sm)]

    def _combine(self, name: str, x, modules: [nn.Module]) -> torch.Tensor:
        if self._fixed.get(name, False):
            return modules[self._fixed.get(name)](x)
        results = [modules[i](x) * w for (i, w) in self._combine_info(name)]
        return sum(results)
