import torch
from uninas.modules.mixed.mixedop import AbstractDependentMixedOp
from uninas.methods.strategy_manager import StrategyManager
from uninas.register import Register


@Register.network_mixed_op()
class SplitWeightsMixedOp(AbstractDependentMixedOp):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine

    in addition, load different sets of weights for the operations,
    depending on architecture choices in previous layers

    due to the used saving/loading approach, this operation will most likely malfunction in distributed settings
    """
    max_depth = 2

    def __init__(self, submodules: list, priors: list, name: str, strategy_name: str, depth=0):
        """
        :param submodules: list or nn.ModuleList of choices
        :param priors: list of indices, which prior candidates to consider for additional super-network weights
        :param name: name of the architecture weight
        :param strategy_name: name of the architecture strategy to use
        :param depth: depth, how many previous architecture decisions to consider
        """
        super().__init__(submodules, priors, name, strategy_name)
        # store previous names in case this mixed op will be deepened, no need to store the own name
        self._add_to_kwargs(depth=depth)
        self._all_prev_names = self._get_prev_names(name, priors, include_self=False)
        self._state_dicts = {}
        self._last_state = 'w'
        self.change_depth(new_depth=self.depth)

    def change_depth(self, new_depth=1):
        """
        called by a SplitWeightsMixedOpCallback,
        increases the recursive depth of the op, copying the weights, using a copy depending on a previous layer choice
        """
        if new_depth > 0:
            assert new_depth >= self.depth, "Can not reduce the depth"
            assert new_depth <= self.max_depth, "Can not increase the depth beyond %d" % self.max_depth
            assert StrategyManager().is_only_single_path()
        while self.depth < min([new_depth, len(self._all_prev_names)]):
            if len(self._state_dicts) == 0:
                self._state_dicts[self._last_state] = self.submodules.state_dict()
            # enlarge dict of stored state dicts by one layer
            new_state_dicts = {'0.%s' % k: v for k, v in self._state_dicts.items()}
            self._state_dicts = new_state_dicts
            self._last_state = '0.%s' % self._last_state
            self.depth += 1
        # print('change %s depth' % self.__class__.__name__, self.name, self.depth)

    def _get_current_state_name(self) -> str:
        """ get a name for the current setting (e.g. "0.1.w") that depends on the previously chosen indices """
        names = self._all_prev_names[-self.depth:]
        return '.'.join([str(self.sm.get_finalized_indices(n, flat=True)) for n in names] + ['w'])

    def _set_weight(self):
        if self.depth > 0:
            # get name of currently used local architecture
            cur_state = self._get_current_state_name()
            if self._last_state != cur_state:
                # store current weights
                self._state_dicts[self._last_state] = {k: v.detach().clone()
                                                       for k, v in self.submodules.state_dict().items()}
                # load data of current weight into the parameter
                self.submodules.load_state_dict(self._state_dicts.get(cur_state, self._state_dicts[self._last_state]))
                self._last_state = cur_state

    def _save_add_dict(self) -> dict:
        """ additional info stored in the save_dict """
        return dict(depth=self.depth, _last_state=self._last_state, _state_dicts=self._state_dicts)

    def _load_add_dict(self, dct: dict):
        """ additional info restored from the save_dict """
        self.depth = dct.get('depth', self.depth)
        self._last_state = dct.get('_last_state', self._last_state)
        self._state_dicts = dct.get('_state_dicts', self._state_dicts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._set_weight()
        return self.ws.combine(self.name, x, self.submodules)
