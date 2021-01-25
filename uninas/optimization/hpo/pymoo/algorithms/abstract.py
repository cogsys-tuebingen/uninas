from pymoo.model.algorithm import Algorithm
from uninas.utils.args import ArgsInterface, Namespace


class AbstractPymooAlgorithm(ArgsInterface):

    @classmethod
    def from_args(cls, args: Namespace) -> Algorithm:
        """ get a parameterized pymoo algorithm """
        raise NotImplementedError
