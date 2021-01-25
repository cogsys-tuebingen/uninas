from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.point_crossover import PointCrossover
from pymoo.operators.integer_from_float_operator import IntegerFromFloatCrossover
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from uninas.utils.args import ArgsInterface, Argument, Namespace
from uninas.register import Register


class AbstractPymooCrossover(ArgsInterface):
    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Crossover:
        raise NotImplementedError


@Register.hpo_pymoo_crossover()
class SbxPymooCrossover(AbstractPymooCrossover):

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('type', default='int', type=str, choices=['int', 'real'], help='?'),
            Argument('prob', default=0.9, type=float, help='?'),
            Argument('eta', default=30, type=int, help='?'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Crossover:
        type_, prob, eta = cls._parsed_arguments(['type', 'prob', 'eta'], args, index=index)
        if type_ == 'int':
            return IntegerFromFloatCrossover(clazz=SimulatedBinaryCrossover, prob=prob, eta=eta)
        elif type_ == 'real':
            return SimulatedBinaryCrossover(prob=prob, eta=eta)
        raise NotImplementedError


@Register.hpo_pymoo_crossover()
class PointPymooCrossover(AbstractPymooCrossover):

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('k', default=1, type=str, choices=[1, 2], help='num crossover points'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Crossover:
        k = cls._parsed_argument('k', args, index=index)
        return PointCrossover(n_points=k)
