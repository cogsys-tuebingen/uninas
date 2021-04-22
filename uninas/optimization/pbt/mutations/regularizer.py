from uninas.optimization.pbt.mutations.abstract import AbstractFloatMultiplierMutation
from uninas.optimization.pbt.response import PbtServerResponse
from uninas.training.regularizers.abstract import AbstractRegularizer
from uninas.utils.args import Argument
from uninas.register import Register


@Register.pbt_mutation()
class RegularizerPbtMutation(AbstractFloatMultiplierMutation):
    """
    mutation of a regularizer
    """

    def __init__(self, p: float, init_factor: float,
                 multiplier_smaller: float, multiplier_larger: float, regularizer_name: str):
        super().__init__(p, init_factor, multiplier_smaller, multiplier_larger)
        self.regularizer_name = regularizer_name

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('regularizer_name', default="DropOutRegularizer", type=str, help="class name of the regularizer to mutate"),
        ]

    def _get_relevant_values(self, copied_log_dict: dict) -> dict:
        """
        perform the mutation of the given training state
        :param copied_log_dict: log dict of the checkpoint that is continued from
        """
        return AbstractRegularizer.filter_values_in_dict(copied_log_dict, self.regularizer_name)

    def _set_mutated_value(self, response: PbtServerResponse, value: float) -> PbtServerResponse:
        """
        perform the mutation of the given training state
        :param response: add the mutation in the response
        :param value: value to set
        """
        response.regularizer_values[self.regularizer_name] = value
        return response
