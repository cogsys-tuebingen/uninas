import torch
from uninas.modules.mixed.bias import BiasMixedOp
from uninas.utils.shape import Shape, ShapeOrList
from uninas.register import Register


@Register.network_mixed_op()
class MulBiasMixedOp(BiasMixedOp):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine

    in addition, add a bias factor depending on the current and previous operation(s)
    """

    def _build(self, s_in: ShapeOrList, c_out: int) -> Shape:
        s_out = super()._build(s_in, c_out)
        if isinstance(self._biases, torch.Tensor):
            self._biases.data += 1
        return s_out

    def _forward_bias(self, x: torch.Tensor) -> torch.Tensor:
        if self._biases is not None:
            prev_arcs = [self.sm.get_finalized_indices(n, flat=True) for n in self._all_names]
            cur_bias = self._biases
            for pa in prev_arcs:
                cur_bias = cur_bias[pa]
            x = x * cur_bias
        return x
