"""

"""


from typing import Union
import torch
import torch.nn as nn
from uninas.models.networks.abstract2 import AbstractExternalNetwork
from uninas.modules.layers.common import LinearLayer, SkipLayer
from uninas.utils.args import Argument
from uninas.utils.shape import Shape, ShapeList
from uninas.utils.misc import split
from uninas.register import Register


@Register.network(external=True)
class FullyConnectedNetwork(AbstractExternalNetwork):
    """
    A fully connected network for vector in- and outputs.
    Uses one activation function after every layer but the last
    """

    def __init__(self, *args, layer_widths: str, act_fun: str, use_bn: bool, use_bias: bool, **kwargs):
        kwargs.update(dict(model_name=self.__class__.__name__))
        super().__init__(*args, **kwargs)
        self._add_to_kwargs(layer_widths=layer_widths, act_fun=act_fun,
                            use_bn=use_bn, use_bias=use_bias)
        self._layer_widths = split(layer_widths, int)
        bias = use_bias and not use_bn

        # first define the network layers, actually build them when we know the input/output sizes
        if len(self._layer_widths) > 0:
            self.stem = LinearLayer(use_bn=use_bn, bn_affine=use_bias, act_fun=act_fun,
                                    act_inplace=False, order='w_bn_act', bias=bias)
        else:
            self.stem = SkipLayer()
        cells = []
        for _ in range(len(self._layer_widths)-1):
            cells.append(LinearLayer(use_bn=use_bn, bn_affine=use_bias, act_fun=act_fun,
                                     act_inplace=False, order='w_bn_act', bias=bias))
        self.cells = nn.ModuleList(cells)
        self.heads = nn.ModuleList([LinearLayer(use_bn=False, act_fun='identity',
                                                act_inplace=False, order='w_bn', bias=True)])

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('layer_widths', default='20, 20', type=str, help='number of features per layer'),
            Argument('act_fun', default='sigmoid', type=str, help='activation function'),
            Argument('use_bn', default='True', type=str, help='use batchnorm', is_bool=True),
            Argument('use_bias', default='True', type=str, help='use bias', is_bool=True),
        ]

    def get_network(self) -> nn.Module:
        return self

    def get_stem(self) -> nn.Module:
        return self.stem

    def get_cells(self) -> nn.ModuleList():
        return self.cells

    def get_heads(self) -> nn.ModuleList():
        return self.heads

    def _build2(self, s_in: Shape, s_out: Shape) -> ShapeList:
        """ build the network """
        assert s_in.num_dims() == s_out.num_dims() == 1
        c_out = self._layer_widths[0] if len(self._layer_widths) > 0 else s_in.num_features()
        s_cur = self.stem.build(s_in, c_out=c_out)
        for i in range(len(self._layer_widths)-1):
            s_cur = self.cells[i].build(s_cur, c_out=self._layer_widths[i+1])
        s_heads = [h.build(s_cur, c_out=s_out.num_features()) for h in self.heads]
        return ShapeList(s_heads)

    def all_forward(self, x: torch.Tensor) -> [torch.Tensor]:
        """
        returns list of all heads' outputs
        the heads are sorted by ascending cell order
        """
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        return [h(x) for h in self.heads]

    def specific_forward(self, x: Union[torch.Tensor, list], start_cell=-1, end_cell=None) -> [torch.Tensor]:
        """
        can execute specific part of the network,
        returns result after end_cell
        """
        if isinstance(x, list):
            x = x[0]

        if start_cell < 0:
            x = self.stem(x)
        if end_cell == -1:
            return [x]

        for i, cell in enumerate(self.cells):
            if start_cell <= i:
                x = cell(x)
            if end_cell == i:
                return [x]

        return [h(x) for h in self.heads]
