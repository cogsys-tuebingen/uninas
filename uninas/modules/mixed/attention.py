import numpy as np
import torch
import torch.nn as nn
from uninas.modules.mixed.mixedop import AbstractDependentMixedOp
from uninas.methods.strategy_manager import StrategyManager
from uninas.utils.shape import Shape, ShapeOrList
from uninas.utils.torch.misc import make_divisible
from uninas.register import Register


class AbstractAttentionMixedOp(AbstractDependentMixedOp):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine

    in addition, apply a channel-weighting, depending on the previous arc choice

    this implementation can nearly double the required training time and has not shown an improvement over the baseline.
    """
    _num_layers = None
    _expansion_width = None
    _expansion_act_fun = None
    _final_act_fun = None
    _include_self = False

    def __init__(self, submodules: list, priors: list, name: str, strategy_name: str):
        """
        :param submodules: list or nn.ModuleList of choices
        :param priors: list of indices, which prior candidates to consider for additional super-network weights
        :param name: name of the architecture weight
        :param strategy_name: name of the architecture strategy to use
        """
        super().__init__(submodules, priors, name, strategy_name)
        # store previous names (and maybe current one), get their number of choices, no need to store the own name
        sm = StrategyManager()
        prev_names = self._get_prev_names(name, priors, self._include_self)
        self._all_prev_sizes = [sm.get_num_weight_choices(n) for n in prev_names]
        self._eye = np.eye(N=max(self._all_prev_sizes + [1]))
        self._attention_op = None
        self._expand_axis = []

    def _build(self, s_in: ShapeOrList, c_out: int) -> Shape:
        s_out = super()._build(s_in, c_out)
        count_in = len(self._all_prev_sizes) * len(self._eye)
        if count_in > 0:
            # fix the size of the input convolution to the maximum possible, to skip size checks at runtime
            assert isinstance(s_in, Shape)
            self._expand_axis = list(range(len(s_in.shape) + 1))
            del self._expand_axis[1]
            # create 1 to n stacked 1x1 conv layers with activation functions in between
            inner_width = make_divisible(count_in * self._expansion_width, divisible=8)
            widths = [count_in] + [inner_width]*(self._num_layers-1) + [s_in.num_features()]
            ops = []
            for i in range(1, len(widths)):
                ops.append(nn.Conv2d(widths[i-1], widths[i], kernel_size=1, bias=True))
                act_fun = self._final_act_fun if (i == len(widths) - 1) else self._expansion_act_fun
                ops.append(Register.act_funs.get(act_fun)())
            self._attention_op = nn.Sequential(*ops)
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
class AttentionSigmoidMixedOp(AbstractAttentionMixedOp):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine

    in addition, apply a channel-weighting, depending on the previous arc choice
    """
    _num_layers = 1
    _expansion_width = 2
    _expansion_act_fun = 'relu'
    _final_act_fun = 'sigmoid'
    _include_self = False


@Register.network_mixed_op()
class AttentionSSigmoidMixedOp(AbstractAttentionMixedOp):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine

    in addition, apply a channel-weighting, depending on the previous arc choice
    """
    _depth = 1
    _num_layers = 1
    _expansion_width = 2
    _expansion_act_fun = 'relu'
    _final_act_fun = 'sigmoid'
    _include_self = True


@Register.network_mixed_op()
class Attention2x1SSigmoidMixedOp(AbstractAttentionMixedOp):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine

    in addition, apply a channel-weighting, depending on the previous arc choice
    """
    _depth = 1
    _num_layers = 2
    _expansion_width = 1
    _expansion_act_fun = 'relu'
    _final_act_fun = 'sigmoid'
    _include_self = True
