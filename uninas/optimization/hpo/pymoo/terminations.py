from pymoo.model.termination import Termination
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
from pymoo.util.termination.max_gen import MaximumGenerationTermination
from pymoo.util.termination.max_time import TimeBasedTermination
from uninas.utils.args import ArgsInterface, Argument, Namespace
from uninas.register import Register


class AbstractPymooTermination(ArgsInterface):
    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Termination:
        raise NotImplementedError


@Register.hpo_pymoo_terminator()
class NEvalPymooTermination(AbstractPymooTermination):
    """
    Stop after 'n' evaluations in total
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('n', default=1000, type=int, help='maximum number function evaluations'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Termination:
        n = cls._parsed_argument('n', args, index=index)
        return MaximumFunctionCallTermination(n)


@Register.hpo_pymoo_terminator()
class NIterPymooTermination(AbstractPymooTermination):
    """
    Stop after 'n' epochs/iterations
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('n', default=100, type=int, help='maximum number algorithm iterations'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Termination:
        n = cls._parsed_argument('n', args, index=index)
        return MaximumGenerationTermination(n)


@Register.hpo_pymoo_terminator()
class TimePymooTermination(AbstractPymooTermination):
    """
    Stop after 's' seconds
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('s', default=3600, type=int, help='maximum algorithm time in seconds'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Termination:
        s = cls._parsed_argument('s', args, index=index)
        return TimeBasedTermination(s)
