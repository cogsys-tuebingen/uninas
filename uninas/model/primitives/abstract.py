from typing import Union
from uninas.model.modules.abstract import AbstractModule
from uninas.model.modules.misc import SequentialModulesF
from uninas.model.modules.mixed import MixedOp
from uninas.model.modules.fused import FusedOp
from uninas.model.layers.common import DifferentConfigLayer
from uninas.utils.misc import split
from uninas.utils.args import ArgsInterface, Argument, Namespace
from uninas.register import Register


class Primitive:
    def instance(self, **layer_kwargs) -> AbstractModule:
        raise NotImplementedError


class DifferentConfigPrimitive(Primitive):
    """ use one primitive for forward passes, but the second one to get the config """

    def __init__(self, forward_primitive: Primitive, config_primitive: Primitive):
        self.forward_primitive = forward_primitive
        self.config_primitive = config_primitive

    def instance(self, **layer_kwargs) -> AbstractModule:
        fm = self.forward_primitive.instance(**layer_kwargs)
        cm = self.config_primitive.instance(**layer_kwargs)
        return DifferentConfigLayer(fm, cm)


class PrimitiveSet(ArgsInterface):
    """ a set of primitives, used in search blocks """

    def __init__(self, strategy_name: str, fused: bool, mixed_cls: str, subset: str):
        super().__init__()
        self.strategy_name = strategy_name
        self.fused = fused
        self.mixed_cls = mixed_cls
        self.subset = split(subset, int)

        if self.fused:
            assert len(self.subset) == 0, "Can not use a subset of ops if the op is fused!"
        else:
            assert (len(self.subset) == 0) or (len(self.subset) >= 2),\
                "The primitives subset must have at least two operations!"

    def instance(self, name: str, **primitive_kwargs) -> AbstractModule:
        if self.fused:
            module = self.fused_instance(name=name, **primitive_kwargs)
            assert isinstance(module, AbstractModule), "fused instance not implemented"
        else:
            module = self.mixed_instance(name=name, **primitive_kwargs)
            assert isinstance(module, AbstractModule), "mixed instance not implemented"
        return module

    @classmethod
    def from_args(cls, args: Namespace, index: int = None) -> 'PrimitiveSet':
        all_parsed = cls._all_parsed_arguments(args, index=index)
        return cls(**all_parsed)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        mixed_ops = Register.network_mixed_ops.names()
        return super().args_to_add(index) + [
            Argument('strategy_name', default="default", type=str, help='under which strategy to register'),
            Argument('fused', default="False", type=str, is_bool=True, help='use a fused operation'),
            Argument('mixed_cls', default=MixedOp.__name__, type=str, choices=mixed_ops, help='class for mixed op'),
            Argument('subset', default="", type=str, help='[int] use only these operations, must not be fused'),
        ]

    def fused_instance(self, name: str, **primitive_kwargs) -> AbstractModule:
        """ get a fused op of all primitives """
        fused = self._fused_instance(name, **primitive_kwargs)
        assert fused is not None, "%s: fused op not defined (is none)" % self.__class__.__name__
        assert isinstance(fused, FusedOp),\
            "%s: fused modules must inherit from %s" % (self.__class__.__name__, FusedOp.__name__)
        assert isinstance(fused, AbstractModule),\
            "%s: fused modules must inherit from %s" % (self.__class__.__name__, AbstractModule.__name__)
        return fused

    def _fused_instance(self, name: str, **primitive_kwargs) -> Union[AbstractModule, None]:
        """ get a fused op of all primitives """
        return None

    def mixed_instance(self, name: str, **primitive_kwargs) -> MixedOp:
        """ get a mixed op of all primitives """
        ops = [p.instance(**primitive_kwargs) for p in self.get_primitives(**primitive_kwargs)]
        assert len(ops) >= 2, "%s has not enough options to choose from" % self.__class__.__name__
        if len(self.subset) > 0:
            ops = [ops[i] for i in self.subset]
        mixed_op_cls = Register.network_mixed_ops.get(self.mixed_cls)
        return mixed_op_cls(submodules=ops, name=name, strategy_name=self.strategy_name)

    @classmethod
    def get_primitives(cls, **primitive_kwargs) -> [Primitive]:
        return []


class CNNPrimitive(Primitive):
    """ a possible cnn operation """

    def __init__(self, cls, args: list = None, kwargs: dict = None, stacked=1):
        self.cls = cls
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.stacked = stacked

    def instance(self, **layer_kwargs) -> AbstractModule:
        kwargs = self.kwargs.copy()
        kwargs.update(layer_kwargs)
        stride = kwargs.pop('stride')
        ops = [self.cls(*self.args, stride=stride if i == 0 else 1, **kwargs) for i in range(self.stacked)]
        if self.stacked == 1:
            return ops[0]
        return SequentialModulesF(ops)


class StrideChoiceCNNPrimitive(Primitive):
    """ choose among primitives based on stride """

    def __init__(self, primitives: [CNNPrimitive]):
        self.primitives = primitives

    def instance(self, **layer_kwargs) -> AbstractModule:
        stride = layer_kwargs.get('stride')
        assert stride is not None
        return self.primitives[stride - 1].instance(**layer_kwargs)
