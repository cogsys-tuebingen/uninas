from pymoo.model.sampling import Sampling
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.operators.integer_from_float_operator import IntegerFromFloatSampling
from uninas.utils.args import ArgsInterface, Namespace
from uninas.register import Register


class AbstractPymooSampler(ArgsInterface):
    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Sampling:
        raise NotImplementedError


@Register.hpo_pymoo_sampler()
class IntRandomPymooSampler(AbstractPymooSampler):
    """
    Sample a random integer (from floats)
    """

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Sampling:
        return IntegerFromFloatSampling(clazz=FloatRandomSampling)


@Register.hpo_pymoo_sampler()
class IntLhsPymooSampler(AbstractPymooSampler):
    """
    Sample a random integer (from latin hyper cube)
    """

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Sampling:
        return IntegerFromFloatSampling(clazz=LatinHypercubeSampling)
