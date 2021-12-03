import torch
import torch.nn as nn
from uninas.modules.heads.abstract import AbstractHead
from uninas.utils.args import Argument
from uninas.utils.shape import Shape
from uninas.register import Register


@Register.network_head()
class PwClassificationHead(AbstractHead):
    """
    Network output, pixel-wise
    act fun, dropout, conv1x1
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('act_fun', default='relu', type=str, help='act fun of the 1x1 convolution', choices=Register.act_funs.names()),
            Argument('bias', default='True', type=str, help='add a bias', is_bool=True),
            Argument('dropout', default=0.0, type=float, help='initial dropout probability'),
        ]

    def set_dropout_rate(self, p=None) -> int:
        if p is not None:
            self.head_module[-2].p = p
            return 1
        return 0

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        ops = [
            Register.act_funs.get(self.act_fun)(inplace=False),
            nn.Dropout(p=self.dropout),
            nn.Conv2d(s_in.num_features(), s_out.num_features(), 1, 1, 0, bias=self.bias),
        ]
        self.head_module = nn.Sequential(*ops)
        return self.probe_outputs(s_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_module(x)
