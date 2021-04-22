import torch
import torch.nn as nn
from uninas.modules.modules.abstract import AbstractModule
from uninas.modules.layers.abstract import AbstractLayer, AbstractStepsLayer
from uninas.utils.shape import Shape
from uninas.register import Register


@Register.network_layer()
class SkipLayer(AbstractLayer):

    def __init__(self, **base_kwargs):
        assert base_kwargs.pop('stride', 1) == 1
        super().__init__(**base_kwargs)

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        feature_diff = c_out - s_in.num_features()
        assert feature_diff >= 0
        self._add_to_print_kwargs(features=c_out, feature_diff=feature_diff)
        return self.probe_outputs(s_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = x.clone()
        if self.feature_diff > 0:
            s = list(x.shape)
            s[1] = self.feature_diff
            x = torch.cat([x, torch.zeros(size=s, device=x.device)], dim=1)
        return x


@Register.network_layer()
class LinearLayer(AbstractStepsLayer):
    dropout_fun = nn.Dropout
    batchnorm_fun = nn.BatchNorm1d

    def __init__(self, bias=False, **base_kwargs):
        super().__init__(**base_kwargs)
        self._add_to_kwargs(bias=bias)

    def _build(self, s_in: Shape, c_out: int, weight_functions=()) -> Shape:
        wf = list(weight_functions) + [nn.Linear(s_in.num_features(), c_out, self.bias)]
        return super()._build(s_in, c_out, weight_functions=wf)


# layers for the search process that will be changed afterwards


@Register.network_layer()
class DifferentConfigLayer(AbstractLayer):
    """
    A wrapper layer that has two inner layers, using one for forward passes, but returning the config of the second one,
    the network_configs are always originals (as this is intended to be used just for a single layer).
    Enables DARTS-like searches without too much tampering (e.g. poolings)
    """

    def __init__(self, forward_module: AbstractModule, config_module: AbstractModule, **__):
        super().__init__(**__)
        self._add_to_submodules(forward_module=forward_module)
        self.config_module = config_module
        self.config_module_cfg = None

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        # get originals config of config_module, then remove the module
        self.config_module.build(s_in, c_out)
        self.config_module_cfg = self.config_module.config(finalize=True)
        del self.config_module
        # forward_module
        return self.forward_module.build(s_in, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_module(x)

    def config(self, **_) -> dict:
        return self.config_module_cfg
