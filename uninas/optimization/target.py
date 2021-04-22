import torch
from uninas.utils.args import ArgsInterface, Argument, Namespace
from uninas.register import Register


@Register.optimization_target()
class OptimizationTarget(ArgsInterface):
    """
    rank networks etc. according to a key in their respective log dicts
    """

    def __init__(self, key: str, maximize: bool):
        super().__init__()
        self.key = key
        self.maximize = maximize

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('key', default="val/accuracy/1", type=str, help="target key in the log dict"),
            Argument('maximize', default="True", type=str, help="maximize or minimize the target?", is_bool=True),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> 'OptimizationTarget':
        return cls(**cls._all_parsed_arguments(args, index=index))

    def get_key(self) -> str:
        return self.key

    def is_maximize(self) -> bool:
        return self.maximize

    def is_interesting(self, log_dict: dict) -> bool:
        """ if the watched key is not in the log dict, no need to synchronize """
        return self.key in log_dict

    def as_str(self, log_dict: dict) -> str:
        v = log_dict.get(self.key)
        if isinstance(v, float):
            s = "%.3f" % v
        else:
            s = str(v)
        return "%s=%s" % (self.key, s)

    def sort_value(self, log_dict: {str: torch.Tensor}) -> float:
        """ get the value to sort, smaller values are better """
        v = float(log_dict[self.key])
        return self.signed_value(v)

    def signed_value(self, v: float) -> float:
        return -v if self.maximize else v
