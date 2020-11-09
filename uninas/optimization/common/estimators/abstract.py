"""
estimators (metrics) to rank different networks (architecture subsets of a supernet)
"""

import numpy as np
from uninas.optimization.hpo_self.candidate import Candidate
from uninas.utils.args import ArgsInterface, Namespace, Argument


class AbstractEstimator(ArgsInterface):
    def __init__(self, args: Namespace, index=None, **kwargs):
        super().__init__()
        all_parsed = self._all_parsed_arguments(args, index)
        self.key = all_parsed.pop('key')
        bounds = (all_parsed.pop('min_value'), all_parsed.pop('max_value'))
        self.bounds = bounds if all_parsed.pop('is_constraint') else None
        self._maximize = all_parsed.pop('maximize')
        self._weighting = all_parsed.pop('weighting', 1.0)
        self._is_objective = all_parsed.pop('is_objective')
        self._sign = -1 if self._maximize else 1
        kwargs.update(all_parsed)
        self.kwargs = kwargs

    def __deepcopy__(self, memo):
        # avoid trainer copy issues
        return self

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            # common
            Argument('key', default='not_set', type=str, help='key (name) of this estimator'),
            # constraints
            Argument('is_constraint', default='False', type=str, help='use as a constraint', is_bool=True),
            Argument('min_value', default=1.0, type=float, help='min allowed value of this constraint'),
            Argument('max_value', default=1.0, type=float, help='max allowed value of this constraint'),
            # objectives
            Argument('is_objective', default='False', type=str, help='use as an objective', is_bool=True),
            Argument('maximize', default='False', type=str, help='maximize this objective', is_bool=True),
        ]

    def is_objective(self) -> bool:
        return self._is_objective

    def is_constraint(self) -> bool:
        return self.bounds is not None

    def name(self, only_minimize=True) -> str:
        if self._maximize and only_minimize:
            return '(-1) * %s' % self.key
        return self.key

    def sign(self, only_minimize=True) -> int:
        if only_minimize:
            return self._sign
        return 1

    def is_allowed(self, candidate: Candidate):
        return self.bounds[0] <= candidate.metrics.get(self.key) <= self.bounds[1]

    def is_dominated(self, candidate1: Candidate, candidate2: Candidate) -> (bool, bool):
        """ check whether candidate1 is dominated by candidate2, or equal """
        if candidate1.id == candidate2.id:
            return False, True
        v1, v2 = candidate1.metrics.get(self.key), candidate2.metrics.get(self.key)
        if self._maximize:
            return v1 <= v2, v1 == v2
        return v1 >= v2, v1 == v2

    def get_ref_point(self, default=0) -> float:
        if self.is_constraint():
            return self.bounds[1] * self._sign
        return default

    def evaluate_candidate(self, candidate: Candidate, strategy_name: str = None) -> float:
        """
        evaluate this metric for the candidate, store info in it
        """
        if self.key not in candidate.metrics:
            candidate.metrics[self.key] = self.evaluate_tuple(candidate.values, strategy_name=strategy_name)
        return candidate.metrics.get(self.key)

    def evaluate_pymoo(self, x: np.array, in_constraints: bool, strategy_name: str = None) -> (np.array, np.array):
        """
        evaluate this metric for the candidate if it is still within constraints,
        return metric value and constraint value
        """
        r, g = self.evaluate_tuple(tuple(x), strategy_name=strategy_name) if in_constraints else 0, None
        if self.is_constraint():
            g = -1 if in_constraints and self.bounds[0] < r < self.bounds[1] else 1
        return self._sign*r, g

    def evaluate_tuple(self, values: tuple, strategy_name: str = None) -> float:
        """
        :param values: architecture description
        :param strategy_name: None if candidate is global, otherwise specific to this weight strategy
        """
        raise NotImplementedError

    def _str_dict(self) -> dict:
        dct = dict(key='"%s"' % self.key)
        if self.is_constraint():
            dct.update(dict(allowed_range=self.bounds))
        if self.is_objective():
            dct.update(dict(maximize=self._maximize, weighting=self._weighting))
        dct.update(self.kwargs)
        return dct
