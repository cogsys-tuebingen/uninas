import torch
from torch import nn as nn
from uninas.modules.modules.abstract import AbstractModule, tensor_type
from uninas.register import Register
from uninas.utils.torch.misc import drop_path
from uninas.utils.shape import Shape, ShapeList, ShapeOrList


class DropPathModule(AbstractModule):
    """
    wrapping drop_path as a module, will not be saved in the config
    designed to work together with DropPathRegularizer
    """

    def __init__(self, module: nn.Module = None, is_skip_module=False, drop_p=0.0, drop_ids=False):
        super().__init__()
        module = module if module is not None else nn.Identity()
        self._add_to_submodules(module=module)
        self._add_to_kwargs(is_skip_module=is_skip_module, drop_p=drop_p, drop_ids=drop_ids)

    def set_drop_ids(self, b: bool):
        self.set(drop_ids=b)

    def set_drop_rate(self, p: float):
        self.set(drop_p=p)

    def _build(self, *args, **kwargs) -> ShapeOrList:
        return self.module.build(*args, **kwargs)

    def forward(self, x: tensor_type) -> tensor_type:
        r = self.module(x)
        if self.training and (self.drop_ids or not self.is_skip_module):
            r = drop_path(r, self.drop_p)
        return r

    def config(self, **_) -> dict:
        return self.module.config(**_)


class MultiModules(AbstractModule):
    """ contains multiple modules, choose from the sequential/parallel classes """
    has_layer_fun = any

    def __init__(self, submodules: [AbstractModule], **__):
        super().__init__(**__)
        submodules = submodules if isinstance(submodules, nn.ModuleList) else nn.ModuleList(submodules)
        self._add_to_submodule_lists(submodules=submodules)

    def set_dropout_rate(self, p=None) -> int:
        n = 0
        for m in self.submodules:
            n += m.set_dropout_rate(p)
        return n

    def config(self, minimize=False, **_) -> dict:
        if len(self.submodules) == 1 and minimize:
            return self.submodules[0].config(minimize=minimize, **_)
        return super().config(minimize=minimize, **_)

    def is_layer(self, cls) -> bool:
        return self.has_layer_fun([m.is_layer(cls) for m in self.submodules]) or super().is_layer(cls)

    def is_layer_path(self, cls) -> [bool]:
        return [m.is_layer(cls) for m in self.submodules]

    def build(self, s_in: ShapeOrList, c_out: int) -> Shape:
        return super().build(s_in, c_out)

    def _build(self, s_in: ShapeOrList, c_out: int) -> Shape:
        raise NotImplementedError

    def forward(self, x: tensor_type) -> tensor_type:
        raise NotImplementedError


class SequentialModules(MultiModules):
    has_layer_fun = any

    def _build(self, s_in: ShapeOrList, c_out: int) -> Shape:
        raise NotImplementedError

    def forward(self, x: tensor_type) -> tensor_type:
        for m in self.submodules:
            x = m(x)
        return x


@Register.network_module()
class SequentialModulesF(SequentialModules):
    """ if c_in != c_out, always correct the difference in the first module """

    def _build(self, s_in: ShapeOrList, c_out: int) -> Shape:
        s = self.submodules[0].build(s_in, c_out)
        for i in range(1, len(self.submodules)):
            s = self.submodules[i].build(s, c_out)
        return s


@Register.network_module()
class SequentialModulesL(SequentialModules):
    """ if c_in != c_out, always correct the difference in the last module """

    def _build(self, s_in: ShapeOrList, c_out: int) -> Shape:
        s = s_in
        num_features = s.num_features() if isinstance(s, Shape) else s[0].num_features()
        for i in range(0, len(self.submodules)-1):
            s = self.submodules[i].build(s, num_features)
        return self.submodules[-1].build(s, c_out)


@Register.network_module()
class ParallelModules(MultiModules):
    has_layer_fun = all

    def _build(self, s_in: ShapeOrList, c_out: int) -> Shape:
        shapes = [self.submodules[i].build(s_in, c_out) for i in range(0, len(self.submodules))]
        for s0, s1 in zip(shapes[:-1], shapes[1:]):
            assert s0 == s1, "shape mismatch: %s, %s" % (str(s0), str(s1))
        return shapes[0]

    def forward(self, x: tensor_type) -> torch.Tensor:
        raise NotImplementedError


@Register.network_module()
class SumParallelModules(ParallelModules):

    def forward(self, x: tensor_type):
        return sum(m(x) for m in self.submodules)


@Register.network_module()
class InputChoiceWrapperModule(AbstractModule):
    """ picks the desired input and returns the output of its wrapped module """

    def __init__(self, wrapped: AbstractModule, idx=0, **__):
        super().__init__(**__)
        self._add_to_submodules(wrapped=wrapped)
        self._add_to_kwargs(idx=idx)

    def is_layer(self, cls) -> bool:
        return isinstance(self, cls) or self.wrapped.is_layer(cls)

    def _build(self, s_ins: ShapeList, *_, **__) -> ShapeOrList:
        return self.wrapped.build(s_ins[self.idx], *_, **__)

    def forward(self, x: [torch.Tensor]) -> tensor_type:
        return self.wrapped(x[self.idx])


@Register.network_module()
class ConcatChoiceModule(AbstractModule):

    def __init__(self, idxs=(0, 1), dim=1, **__):
        super().__init__(**__)
        self._add_to_kwargs(idxs=idxs, dim=dim)

    @property
    def num(self):
        return len(self.idxs)

    def _build(self, s_ins: ShapeList, c_out: int) -> Shape:
        chosen = [s for i, s in enumerate(s_ins) if i in self.idxs]
        one = chosen[0].copy()
        c = sum(s[self.dim] for s in chosen)
        one[self.dim] = c
        return one

    def forward(self, x: [torch.Tensor]) -> torch.Tensor:
        # replaces standard forward of BaseLayer, therefore no dropout/bn
        selected = [x[idx] for idx in self.idxs]
        return torch.cat(selected, dim=self.dim)
