from uninas.modules.modules.abstract import AbstractModule
from uninas.modules.modules.misc import SequentialModulesF
from uninas.modules.mixed.mixedop import MixedOp
from uninas.modules.layers.common import DifferentConfigLayer
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

    def __init__(self, strategy_name: str, mixed_cls: str, mixed_priors: str, subset: str):
        super().__init__()
        self.strategy_name = strategy_name
        self.mixed_cls = mixed_cls
        self.mixed_priors = -1 if len(mixed_priors) == 0 else eval(mixed_priors)
        self.subset = split(subset, int)
        self._num_instances = 0

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
            Argument('mixed_cls', default=MixedOp.__name__, type=str, choices=mixed_ops, help='class for mixed op'),
            Argument('mixed_priors', default="", type=str,
                     help='empty string: consider only the immediately prior candidate op; '
                          'or str([[int]]): which prior candidates to consider (by their indices)'),
            Argument('subset', default="", type=str, help='[int] use only these operations, must not be fused'),
        ]

    def instance(self, name: str, **primitive_kwargs) -> MixedOp:
        """ get a mixed op of all primitives """
        # get all primitives, make sure only one kind is implemented
        ops1 = [p.instance(**primitive_kwargs) for p in self.get_primitives(**primitive_kwargs)]
        ops2 = self.get_shared_instance_primitives(name, self.strategy_name, **primitive_kwargs)
        available_ops = [ops1, ops2]
        lengths = [len(ops) for ops in available_ops]
        assert sum(lengths) == max(lengths), "%s can only have one kind of implementation" % self.__class__.__name__

        # pick the correct array
        ops = None
        for ops_ in available_ops:
            if len(ops_) > 0:
                ops = ops_
                break
        assert len(ops) >= 2, "%s has not enough options to choose from" % self.__class__.__name__

        # maybe only use a subset
        if len(self.subset) > 0:
            ops = [ops[i] for i in self.subset]

        # if inter-candidate dependent weights are used, which prior candidates to consider
        if isinstance(self.mixed_priors, int):
            priors = [self.mixed_priors]
        elif isinstance(self.mixed_priors, list):
            priors = self.mixed_priors[self._num_instances % len(self.mixed_priors)]
        else:
            raise NotImplementedError
        assert isinstance(priors, list)
        assert all([isinstance(p, int) for p in priors])

        # create mixed op
        self._num_instances += 1
        mixed_op_cls = Register.network_mixed_ops.get(self.mixed_cls)
        return mixed_op_cls(submodules=ops, priors=priors, name=name, strategy_name=self.strategy_name)

    @classmethod
    def get_primitives(cls, **primitive_kwargs) -> [Primitive]:
        return []

    @classmethod
    def get_shared_instance_primitives(cls, name: str, strategy_name: str, **primitive_kwargs) -> [AbstractModule]:
        return []


class CNNPrimitive(Primitive):
    """ a possible cnn operation """

    def __init__(self, cls, args: list = None, kwargs: dict = None, stacked=1):
        self.cls = cls
        self.args = args if args is not None else []
        self.kwargs = dict(stride=1)
        if isinstance(kwargs, dict):
            self.kwargs.update(kwargs)
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
