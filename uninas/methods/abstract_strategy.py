from collections import defaultdict, Sequence

import torch
import torch.nn as nn
from uninas.utils.args import ArgsInterface
from uninas.register import Register


class RequestedWeight:
    """
    helper class to request weights and add the to the strategy later
    """

    def __init__(self, name: str):
        self.name = name
        self.requested_by = []
        self._num_requests = 0
        self._choice_indices = []
        self._masked_choice_indices = set()

    def add_request(self, choices: nn.ModuleList = None, num_choices: int = None):
        assert isinstance(choices, nn.ModuleList) or isinstance(num_choices, int)
        num_choices = len(choices) if choices is not None else num_choices
        if len(self.requested_by) > 0:
            assert self.num_choices() == num_choices,\
                "Multiple weights with the same name need to have the same number of choices!"
        self._choice_indices = list(range(num_choices))
        self._num_requests += 1
        if isinstance(choices, nn.ModuleList):
            self.requested_by.append(choices)

    def num_requests(self) -> int:
        return self._num_requests

    def num_choices(self) -> int:
        return len(self._choice_indices)

    def num_choices_str(self) -> str:
        if len(self._masked_choice_indices) > 0:
            return "%d (of %d)" % (self.num_choices(), self.num_choices() + len(self._masked_choice_indices))
        return str(self.num_choices())

    def get_choices(self) -> [int]:
        return self._choice_indices.copy()

    def mask_index(self, idx: int):
        self._choice_indices.remove(idx)
        self._masked_choice_indices.add(idx)


class AbstractWeightStrategy(nn.Module, ArgsInterface):
    """
    Storing architecture weights (if any) and defining how they are used in super-network forward passes
    """

    def __init__(self, max_epochs: int, name='default'):
        super().__init__()
        self.max_epochs = max_epochs
        self.name = name

        self._requested = {}            # {name: RequestedWeight}
        self._ordered = []              # RequestedWeights in order of requests
        self._ordered_unique = []       # RequestedWeights in order of requests, ignoring duplicates

        self._losses = defaultdict(list)

    def get_name(self) -> str:
        return self.name

    def on_epoch_start(self, current_epoch: int):
        """ whenever the method starts a new epoch """
        pass

    def on_epoch_end(self, current_epoch: int) -> bool:
        """
        whenever the method ends an epoch
        signal early stopping when returning True
        """
        return False

    def _requested_sizes(self) -> {int: list}:
        sizes = defaultdict(list)
        for r in self._ordered_unique:
            sizes[r.num_choices()].append(r)
        return sizes

    def str(self) -> str:
        return '%s("%s", %d architecture weights)' % (self.__class__.__name__, self.name, len(self._ordered_unique))

    def get_log_dict(self) -> {str: float}:
        """
        :return: dict of values that are interesting to log
        """
        return {}

    def make_weight(self, name: str, choices: nn.ModuleList = None, num_choices: int = None):
        """
        register that a parameter of given name and num choices will be required during the search
        called by network components before the network is built
        """
        num_choices = len(choices) if choices is not None else num_choices
        if num_choices <= 0:
            return None
        if self._requested.get(name, None) is None:
            self._requested[name] = RequestedWeight(name)
            self._ordered_unique.append(self._requested[name])
        self._requested[name].add_request(choices, num_choices)
        self._ordered.append(self._requested[name])

    def max_num_choices(self) -> int:
        return max([r.num_choices() for r in self._ordered_unique])

    def get_num_choices(self) -> [int]:
        """
        num choices per weight, by request order
        """
        return [r.num_choices() for r in self._ordered_unique]

    def ordered_names(self, unique=True) -> [str]:
        """
        name of each weight, by request order
        """
        lst = self._ordered_unique if unique else self._ordered
        return [r.name for r in lst]

    def get_weight_names(self) -> [str]:
        return [r.name for r in self._ordered_unique]

    def get_requested_weight(self, name: str) -> RequestedWeight:
        return self._requested.get(name)

    def get_requested_weights(self) -> [RequestedWeight]:
        return self._ordered_unique

    def randomize_weights(self):
        """ randomizes all arc weights """
        raise NotImplementedError

    def build(self):
        """
        actually generate all the requested weights, group weights with same num params for efficiency
        called once after the network is built, only then it has requested all required architecture weights
        """
        raise NotImplementedError

    def forward_const(self, const=0):
        """
        forward pass, setting fixed_arc to a const value
        """
        self.forward(fixed_arc=[const]*len(self._ordered_unique))

    def forward(self, fixed_arc=None, **__):
        """
        called once before every network forward pass
        pre-compute values of interest, e.g. softmax over weights
        """
        raise NotImplementedError

    def get_weight_sm(self, name: str) -> torch.Tensor:
        """ softmax over the specified weight """
        raise NotImplementedError

    def get_finalized_index(self, name: str) -> int:
        """ return index of the module that should constitute the new architecture, for this specific weight """
        indices = self.get_finalized_indices(name)
        assert len(indices) == 1, "Have %d indices but must select one, which is ambiguous" % len(indices)
        return indices[0]

    def get_finalized_indices(self, name: str) -> [int]:
        """ return indices of the modules that should constitute the new architecture, for this specific weight """
        raise NotImplementedError

    def combine_info(self, name: str) -> Sequence:
        """ get a tuple or list of (idx, weight), instructing how to sum the path modules """
        try:
            return self._combine_info(name)
        except:
            return (0, 1.0),

    def _add_loss(self, name: str, loss: torch.Tensor):
        """ add a loss tensor to the stored losses """
        self._losses[name].append(loss)

    def get_losses(self, clear=True) -> {str, torch.Tensor}:
        """ get loss tensors, maybe clear storage """
        losses = {k: sum(v) for k, v in self._losses.items()}
        if clear:
            self._losses.clear()
        return losses

    def _combine_info(self, name: str) -> Sequence:
        """ get a tuple or list of (idx, weight), instructing how to sum the path modules """
        raise NotImplementedError

    def combine(self, name: str, x, modules: [nn.Module]) -> torch.Tensor:
        """
        combine multiple outputs into one, depending on arc weights

        :param name: name of the SearchModule object
        :param x: input (e.g. torch.Tensor)
        :param modules: torch.nn.Modules, may be None if module_results are available
        :return: combination of module results
        """
        raise NotImplementedError

    def mask_index(self, idx: int, weight_name: str = None):
        """ prevent sampling a specific index, either for a specific weight or all weights """
        weight_names = [weight_name] if isinstance(weight_name, str) else self.get_weight_names()
        for n in weight_names:
            self._mask_index(idx, n)
            self.get_requested_weight(n).mask_index(idx)

    def _mask_index(self, idx: int, weight_name: str):
        raise NotImplementedError

    def feedback(self, key: str, log_dict: dict, current_epoch: int, batch_idx: int):
        """
        feedback after each training forward pass
        this is currently not synchronized in distributed training

        :param key: train/val/test
        :param log_dict: contains loss, metric results, ...
        :param current_epoch:
        :param batch_idx:
        :return:
        """
        pass

    @classmethod
    def is_single_path(cls) -> bool:
        return Register.get_my_kwargs(cls).get('single_path', False)
