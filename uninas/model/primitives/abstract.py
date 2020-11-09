from uninas.model.modules.abstract import AbstractModule
from uninas.model.modules.misc import SequentialModulesF, MixedOp
from uninas.model.layers.common import DifferentConfigLayer


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


class PrimitiveSet:
    """ a set of primitives, used in search blocks """

    @classmethod
    def mixed_instance(cls, name: str, strategy_name='default', **primitive_kwargs) -> MixedOp:
        """ get a mixed op of all primitives """
        ops = [p.instance(**primitive_kwargs) for p in cls._primitives()]
        return MixedOp(submodules=ops, name=name, strategy_name=strategy_name)

    @classmethod
    def _primitives(cls, **primitive_kwargs) -> [Primitive]:
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
