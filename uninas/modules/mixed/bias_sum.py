import torch
import torch.nn as nn
from uninas.modules.mixed.mixedop import AbstractDependentMixedOp
from uninas.methods.strategy_manager import StrategyManager
from uninas.utils.shape import Shape, ShapeOrList
from uninas.register import Register


@Register.network_mixed_op()
class BiasSumMixedOp(AbstractDependentMixedOp):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine

    in addition, add a bias depending on the current and previous operation(s)
    """

    def __init__(self, submodules: list, priors: list, name: str, strategy_name: str):
        """
        :param submodules: list or nn.ModuleList of choices
        :param priors: list of indices, which prior candidates to consider for additional super-network weights
        :param name: name of the architecture weight
        :param strategy_name: name of the architecture strategy to use
        """
        super().__init__(submodules, priors, name, strategy_name)
        # store previous names and current one, get their number of choices, no need to store the own name
        sm = StrategyManager()
        self._all_names = self._get_prev_names(name, priors, include_self=True)
        self._all_sizes = [sm.get_num_weight_choices(n) for n in self._all_names]
        self._biases = None

    def _build(self, s_in: ShapeOrList, c_out: int) -> Shape:
        s_out = super()._build(s_in, c_out)
        if len(self._all_sizes) > 1:
            assert isinstance(s_in, Shape)
            biases = []
            # data shape for the bias vector, avoid expand_as at runtime
            last_shape_part = [1] + [1 for _ in range(len(s_in.shape))]
            last_shape_part[1] = s_in.num_features()
            # one bias per previous weight
            for size in self._all_sizes:
                bias_shape = [size] + last_shape_part
                biases.append(nn.Parameter(torch.zeros(size=bias_shape, dtype=torch.float32), requires_grad=True))
            self._biases = nn.ParameterList(biases)
        return s_out

    def _forward_bias(self, x: torch.Tensor) -> torch.Tensor:
        if self._biases is not None:
            for i, name in enumerate(self._all_names):
                idx = self.sm.get_finalized_indices(name, flat=True)
                x = x + self._biases[i][idx]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_bias(x)
        return self.ws.combine(self.name, x, self.submodules)
