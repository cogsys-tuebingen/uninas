import random
import numpy as np
from collections import defaultdict
from uninas.optimization.pbt.response import PbtServerResponse
from uninas.utils.args import ArgsInterface, Argument, Namespace


class AbstractPbtMutation(ArgsInterface):
    """
    mutation of the training state
    """

    def __init__(self, p: float, init_factor: float):
        super().__init__()
        self._values = defaultdict(dict)
        self.p = p
        self.init_factor = init_factor

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('p', default=1.0, type=float, help="probability to mutate"),
            Argument('init_factor', default=1.0, type=float, help="get initial values between v/init and v*init"),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> 'AbstractPbtMutation':
        return cls(**cls._all_parsed_arguments(args, index=index))

    def reset(self):
        """ called before the selection/mutation """
        self._values = defaultdict(dict)

    def pick_value(self, original_value, value_options: list):
        """ try to avoid sampling a value multiple times, pick the one sampled the least often """
        numbers = [self._values[original_value].get(v, 0) for v in value_options]
        m = min(numbers)
        idx = [i for i, n in enumerate(numbers) if n == m]
        x = value_options[random.choices(idx, k=1)[0]]
        self._values[original_value][x] = self._values.get(original_value).get(x, 0) + 1
        return x

    def initial_mutate(self, response: PbtServerResponse, log_dict: dict, num_clients: int) -> (dict, PbtServerResponse):
        """
        perform the initial mutation of the given training state
        :param response: add the mutation in the response
        :param log_dict: log dict
        :param num_clients: number of clients in the task
        :return the changed log dict and the proper response
        """
        raise NotImplementedError

    def mutate(self, response: PbtServerResponse, copied_log_dict: dict) -> PbtServerResponse:
        """
        perform the mutation of the given training state
        :param response: add the mutation in the response
        :param copied_log_dict: log dict of the checkpoint that is continued from
        """
        if random.random() < self.p:
            return self._mutate(response, copied_log_dict)
        return response

    def _mutate(self, response: PbtServerResponse, copied_log_dict: dict) -> PbtServerResponse:
        """
        perform the mutation of the given training state
        :param response: add the mutation in the response
        :param copied_log_dict: log dict of the checkpoint that is continued from
        """
        raise NotImplementedError

    def _set_mutated_value(self, response: PbtServerResponse, value) -> PbtServerResponse:
        """
        perform the mutation of the given training state
        :param response: add the mutation in the response
        :param value: value to set
        """
        raise NotImplementedError


class AbstractFloatMultiplierMutation(AbstractPbtMutation):
    """
    mutation of a float value
    """

    def __init__(self, p: float, init_factor: float, multiplier_smaller: float, multiplier_larger: float):
        super().__init__(p, init_factor)
        self.multiplier_smaller = multiplier_smaller
        self.multiplier_larger = multiplier_larger

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('multiplier_smaller', default=0.8, type=float, help=""),
            Argument('multiplier_larger', default=1.2, type=float, help=""),
        ]

    def _pick_value(self, original_value: float) -> float:
        assert isinstance(original_value, float), "expected float original value, got %s" % str(type(original_value))
        value_options = [original_value * self.multiplier_smaller, original_value * self.multiplier_larger]
        value = self.pick_value(original_value, value_options)
        assert isinstance(value, float), "expected float value, got %s" % str(type(value))
        return value

    def initial_mutate(self, response: PbtServerResponse, log_dict: dict, num_clients: int) -> (dict, PbtServerResponse):
        """
        perform the initial mutation of the given training state
        :param response: add the mutation in the response
        :param log_dict: log dict
        :param num_clients: number of clients in the task
        :return the changed log dict and the proper response
        """
        if num_clients <= 1:
            return log_dict, response

        relevant_dict = self._get_relevant_values(log_dict)
        assert len(relevant_dict) == 1, "Ambiguous or no values (%s): (%s)" % (str(log_dict), str(relevant_dict))
        key = list(relevant_dict.keys())[0]
        value = list(relevant_dict.values())[0]
        assert isinstance(value, float), "expected float original value, got %s" % str(type(value))

        upper = value * self.init_factor
        lower = value / self.init_factor
        value_options = np.arange(0, num_clients) / (num_clients - 1)
        value_options *= (upper - lower)
        value_options += lower

        value = self.pick_value("init", value_options)
        assert isinstance(value, float), "expected float value, got %s" % str(type(value))

        ld = log_dict.copy()
        ld[key] = value
        return ld, self._set_mutated_value(response, value)

    def _mutate(self, response: PbtServerResponse, copied_log_dict: dict) -> PbtServerResponse:
        """
        perform the mutation of the given training state
        :param response: add the mutation in the response
        :param copied_log_dict: log dict of the checkpoint that is continued from
        """
        relevant_dict = self._get_relevant_values(copied_log_dict)
        assert len(relevant_dict) == 1, "Ambiguous or no values (%s): (%s)" % (str(copied_log_dict), str(relevant_dict))
        value = self._pick_value(list(relevant_dict.values())[0])
        return self._set_mutated_value(response, value)

    def _get_relevant_values(self, copied_log_dict: dict) -> dict:
        """
        perform the mutation of the given training state
        :param copied_log_dict: log dict of the checkpoint that is continued from
        """
        raise NotImplementedError

    def _set_mutated_value(self, response: PbtServerResponse, value: float) -> PbtServerResponse:
        """
        perform the mutation of the given training state
        :param response: add the mutation in the response
        :param value: value to set
        """
        raise NotImplementedError
