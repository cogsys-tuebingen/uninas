import numpy as np
import torch
import torch.nn as nn
from uninas.model.modules.misc import SumParallelModules
from uninas.methods.strategies.manager import StrategyManager
from uninas.utils.shape import Shape, ShapeOrList
from uninas.register import Register


@Register.network_mixed_op()
class MixedOp(SumParallelModules):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine
    """

    def __init__(self, submodules: list, name: str, strategy_name: str):
        """
        :param submodules: list or nn.ModuleList of choices
        :param name: name of the architecture weight
        :param strategy_name: name of the architecture strategy to use
        """
        super().__init__(submodules)
        self._add_to_kwargs(name=name, strategy_name=strategy_name)
        self.sm = StrategyManager()
        self.ws = self.sm.make_weight(self.strategy_name, name, only_single_path=False, choices=self.submodules)

    def config(self, finalize=True, **_) -> dict:
        if finalize:
            indices = self.ws.get_finalized_indices(self.name)
            if len(indices) == 1:
                return self.submodules[indices[0]].config(finalize=finalize, **_)
            return SumParallelModules([self.submodules[i] for i in indices]).config(finalize=finalize, **_)
        else:
            return super().config(finalize=finalize, **_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ws.combine(self.name, x, self.submodules)


class AbstractDependentMixedOp(MixedOp):
    """
    a mixed op that somehow depends on previously chosen mixed ops
    """

    def _save_to_state_dict(self, destination: dict, prefix: str, keep_vars: bool):
        # save additional info
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination['%s@added_state' % prefix] = self._save_add_dict()

    def _load_from_state_dict(self, state_dict: dict, prefix: str, local_metadata, strict,
                              missing_keys: list, unexpected_keys: list, error_msgs: list):
        # load additional info
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        added_state = state_dict.get('%s@added_state' % prefix, {})
        if '%s@added_state' % prefix in unexpected_keys:
            unexpected_keys.remove('%s@added_state' % prefix)
        self._load_add_dict(added_state)

    def _save_add_dict(self) -> dict:
        """ additional info stored in the save_dict """
        return {}

    def _load_add_dict(self, dct: dict):
        """ additional info restored from the save_dict """
        pass


class AbstractAttentionMixedOp(AbstractDependentMixedOp):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine

    in addition, apply a channel-weighting, depending on the previous arc choice
    """
    _depth = None
    _act_fun = None

    def __init__(self, submodules: list, name: str, strategy_name: str):
        """
        :param submodules: list or nn.ModuleList of choices
        :param name: name of the architecture weight
        :param strategy_name: name of the architecture strategy to use
        """
        assert None not in [self._depth, self._act_fun], "this class should not be initialized directly"
        super().__init__(submodules, name, strategy_name)
        # store previous names, get their number of choices, no need to store the own name
        sm = StrategyManager()
        self._all_prev_names = sm.ordered_names(unique=False)[-self._depth - 1:-1]
        self._all_prev_sizes = [sm.get_num_weight_choices(n) for n in self._all_prev_names]
        self._eye = np.eye(N=max(self._all_prev_sizes + [1]))
        self._attention_op = None
        self._expand_axis = []

    def _build(self, s_in: ShapeOrList, c_out: int) -> Shape:
        s_out = super()._build(s_in, c_out)
        count = len(self._all_prev_sizes) * len(self._eye)
        if count > 0:
            # fix the size of the convolution to the maximum possible, to skip size checks at runtime
            assert isinstance(s_in, Shape)
            self._expand_axis = list(range(len(s_in.shape) + 1))
            del self._expand_axis[1]
            op1 = nn.Conv2d(count, s_in.num_features(), kernel_size=1, bias=True)
            op2 = Register.act_funs.get(self._act_fun)()
            self._attention_op = nn.Sequential(op1, op2)
        return s_out

    def _forward_attention(self, x: torch.Tensor) -> torch.Tensor:
        if self._attention_op:
            prev_arcs = [self.sm.get_finalized_indices(n, flat=True) for n in self._all_prev_names]
            one_hot = np.concatenate([self._eye[p] for p in prev_arcs], axis=0)
            one_hot = np.expand_dims(one_hot, axis=self._expand_axis)
            one_hot = torch.from_numpy(one_hot).to(x.dtype).to(x.device)
            weighting = self._attention_op(one_hot).expand_as(x)
            x = x * weighting
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_attention(x)
        return self.ws.combine(self.name, x, self.submodules)


@Register.network_mixed_op()
class AttentionD1SigmoidMixedOp(AbstractAttentionMixedOp):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine

    in addition, apply a channel-weighting, depending on the previous arc choice
    """
    _depth = 1
    _act_fun = 'sigmoid'


@Register.network_mixed_op()
class AttentionD2SigmoidMixedOp(AbstractAttentionMixedOp):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine

    in addition, apply a channel-weighting, depending on the previous two arc choices
    """
    _depth = 2
    _act_fun = 'sigmoid'


@Register.network_mixed_op()
class AttentionD3SigmoidMixedOp(AbstractAttentionMixedOp):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine

    in addition, apply a channel-weighting, depending on the previous two arc choices
    """
    _depth = 3
    _act_fun = 'sigmoid'


@Register.network_mixed_op()
class VariableDepthMixedOp(AbstractDependentMixedOp):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine

    in addition, load different sets of weights for the operations,
    depending on architecture choices in previous layers
    """
    max_depth = 2

    def __init__(self, submodules: list, name: str, strategy_name: str, depth=0):
        """
        :param submodules: list or nn.ModuleList of choices
        :param name: name of the architecture weight
        :param strategy_name: name of the architecture strategy to use
        :param depth: depth, how many previous architecture decisions to consider
        """
        super().__init__(submodules, name, strategy_name)
        # store previous names in case this mixed op will be deepened, no need to store the own name
        self._add_to_kwargs(depth=depth)
        self._all_prev_names = StrategyManager().ordered_names(unique=False)[-self.max_depth - 1:-1]
        self._state_dicts = {}
        self._last_state = 'w'
        self.change_depth(new_depth=self.depth)

    def change_depth(self, new_depth=1):
        """
        called by a VariableDepthMixedOpCallback,
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
