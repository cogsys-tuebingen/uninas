import torch
from uninas.modules.heads.abstract import AbstractHead
from uninas.utils.shape import Shape
from uninas.register import Register


@Register.network_head()
class IdentityHead(AbstractHead):
    """
    Network output
    """

    def set_dropout_rate(self, p=None):
        pass

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        assert s_in == s_out
        return self.probe_outputs(s_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
