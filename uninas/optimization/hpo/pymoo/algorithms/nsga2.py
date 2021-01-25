from pymoo.model.algorithm import Algorithm
from pymoo.algorithms.nsga2 import NSGA2

from uninas.utils.args import Argument, Namespace, MetaArgument
from uninas.optimization.hpo.pymoo.algorithms.abstract import AbstractPymooAlgorithm
from uninas.register import Register


@Register.hpo_pymoo_algorithm()
class NSGA2PymooAlgorithm(AbstractPymooAlgorithm):
    """
    NSGA-II (non-dominant sorting genetic algorithm 2)
    https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf

    Principles:
        1) elitism, the best individuals get a chance to reproduce
        2) crowding distance, preserving diversity in the population
        3) emphasizing non-dominated solutions
    """

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        return [
            MetaArgument('cls_hpo_pymoo_sampler', Register.hpo_pymoo_samplers, help_name='sampling method', allowed_num=1),
            MetaArgument('cls_hpo_pymoo_crossover', Register.hpo_pymoo_crossovers, help_name='crossover method', allowed_num=1),
            MetaArgument('cls_hpo_pymoo_mutation', Register.hpo_pymoo_mutations, help_name='mutation method', allowed_num=1),
        ]

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('pop_size', default=5, type=int, help='population size'),
            Argument('n_offsprings', default=10, type=int, help='num offspring per generation'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> Algorithm:
        """ get a parameterized pymoo algorithm """
        return NSGA2(
            pop_size=cls._parsed_argument('pop_size', args, index=index),
            n_offsprings=cls._parsed_argument('n_offsprings', args, index=index),
            sampling=cls._parsed_meta_argument(Register.hpo_pymoo_samplers, 'cls_hpo_pymoo_sampler', args, index).from_args(args),
            crossover=cls._parsed_meta_argument(Register.hpo_pymoo_crossovers, 'cls_hpo_pymoo_crossover', args, index).from_args(args),
            mutation=cls._parsed_meta_argument(Register.hpo_pymoo_mutations, 'cls_hpo_pymoo_mutation', args, index).from_args(args),
            eliminate_duplicates=True,
            save_history=False,
        )
