from uninas.optimization.pbt.mutations.abstract import AbstractFloatMultiplierMutation
from uninas.optimization.pbt.response import PbtServerResponse
from uninas.training.optimizers.abstract import WrappedOptimizer
from uninas.utils.args import Argument
from uninas.register import Register


@Register.pbt_mutation()
class OptimizerPbtMutation(AbstractFloatMultiplierMutation):
    """
    mutation of the learning rate
    """

    def __init__(self, p: float, init_factor: float,
                 multiplier_smaller: float, multiplier_larger: float, optimizer_index: int):
        super().__init__(p, init_factor, multiplier_smaller, multiplier_larger)
        self.optimizer_index = optimizer_index

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('optimizer_index', default=0, type=int, help="index of the optimizer to mutate"),
        ]

    def _get_relevant_values(self, copied_log_dict: dict) -> dict:
        """
        perform the mutation of the given training state
        :param copied_log_dict: log dict of the checkpoint that is continued from
        """
        return WrappedOptimizer.filter_values_in_dict(copied_log_dict, self.optimizer_index)

    def _set_mutated_value(self, response: PbtServerResponse, value: float) -> PbtServerResponse:
        """
        perform the mutation of the given training state
        :param response: add the mutation in the response
        :param value: value to set
        """
        response.optimizer_lrs[self.optimizer_index] = value
        return response
