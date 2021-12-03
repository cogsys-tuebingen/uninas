import torch
from uninas.modules.heads.abstract import AbstractHead
from uninas.utils.shape import Shape
from uninas.register import Register


@Register.network_head()
class IdentityHead(AbstractHead):
    """
    Network output,
    able to perform simple reshaping/squeezing
    """

    def __init__(self, *_, **kwargs_to_store):
        super().__init__(*_, **kwargs_to_store)
        self._reshape = None

    def set_dropout_rate(self, p=None) -> int:
        return 0

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        if s_in != s_out:
            assert s_in.numel() == s_out.numel()
            self._reshape = [-1] + s_out.shape
        return self.probe_outputs(s_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._reshape is not None:
            x = x.reshape(self._reshape)
        return x
