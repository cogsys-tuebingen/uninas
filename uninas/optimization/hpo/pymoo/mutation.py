from pymoo.model.mutation import Mutation
from pymoo.operators.mutation.no_mutation import NoMutation
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.integer_from_float_operator import IntegerFromFloatMutation
from uninas.utils.args import ArgsInterface, Argument, Namespace
from uninas.register import Register


class AbstractPymooCrossover(ArgsInterface):
    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Mutation:
        raise NotImplementedError


@Register.hpo_pymoo_crossover()
class NoPymooMutation(AbstractPymooCrossover):
    """
    No mutation at all
    """

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Mutation:
        return NoMutation()


@Register.hpo_pymoo_mutation()
class PolynomialPymooMutation(AbstractPymooCrossover):
    """
    IntegerFromFloatMutation for integer variables, PolynomialMutation for floats
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('type', default='int', type=str, choices=['int', 'real'], help='?'),
            Argument('eta', default=30, type=int, help='?'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Mutation:
        type_, eta = cls._parsed_arguments(['type', 'eta'], args, index=index)
        if type_ == 'int':
            return IntegerFromFloatMutation(clazz=PolynomialMutation, eta=eta)
        elif type_ == 'real':
            return PolynomialMutation(eta=eta)
        raise NotImplementedError
