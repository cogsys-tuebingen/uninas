"""
estimators (metrics) to rank different networks (architecture subsets of a supernet)
"""

from typing import Union
import numpy as np
from uninas.optimization.hpo.uninas.candidate import Candidate
from uninas.utils.args import ArgsInterface, Namespace, Argument


class AbstractEstimator(ArgsInterface):
    def __init__(self, args: Namespace, index=None, **kwargs):
        super().__init__()
        all_parsed = self._all_parsed_arguments(args, index)
        self.key = all_parsed.pop('key')
        bounds = (all_parsed.pop('min_value'), all_parsed.pop('max_value'))
        self._is_constraint = all_parsed.pop('is_constraint')
        self.bounds = bounds if self._is_constraint else None
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
        return self._is_constraint

    def is_maximize(self) -> bool:
        return self._maximize

    def name(self, only_minimize=True) -> str:
        if self._maximize and only_minimize:
            return '(-1) * %s' % self.key
        return self.key

    def sign(self, only_minimize=True) -> int:
        if only_minimize:
            return self._sign
        return 1

    def is_candidate_allowed(self, candidate: Candidate):
        return self.is_in_constraints(candidate.metrics.get(self.key))

    def is_in_constraints(self, value: float) -> bool:
        if self._is_constraint:
            return self.bounds[0] <= value <= self.bounds[1]
        return True

    def get_constraint_badness(self, value: float) -> float:
        """ get a number how much 'value' violates the constraints, 0 if it is within """
        if self.is_in_constraints(value):
            return 0
        if value < self.bounds[0]:
            return self.bounds[0] / (value + 1e-10)
        return value / self.bounds[1]

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

    def _str_dict(self) -> dict:
        dct = dict(key='"%s"' % self.key)
        if self.is_constraint():
            dct.update(dict(allowed_range=self.bounds))
        if self.is_objective():
            dct.update(dict(maximize=self._maximize, weighting=self._weighting))
        dct.update(self.kwargs)
        return dct

    def evaluate_pymoo(self, x: np.array, in_constraints: np.array, strategy_name: str = None)\
            -> (np.array, Union[np.array, None]):
        """
        evaluate this metric for the candidate if it is still within constraints,
        return metric value and constraint value

        :param x: [batch, ...] numpy array
        :param in_constraints: [bool] numpy array
        :param strategy_name: None if candidate is global, otherwise specific to this weight strategy
        :return:
        """
        # only use the relevant parameters, that are in the hard constraints
        indices = np.argwhere(in_constraints).squeeze()
        x_sub = x[indices]

        # evaluate results
        r = np.zeros(shape=(x.shape[0],), dtype=np.float32)
        if indices.shape[0] > 0:
            r_sub = self.evaluate_batch(x_sub, only_minimize=True, strategy_name=strategy_name)
            r[indices] = r_sub.squeeze()

        # constraints
        c = None
        if self.is_constraint():
            c = np.ones(shape=(x.shape[0],), dtype=np.bool)
            if indices.shape[0] > 0:
                if self.is_constraint():
                    for i in range(len(c)):
                        y = self.bounds[0] < r[i]*self._sign < self.bounds[1]
                        c[i] &= y
            else:
                c &= False

        return r, c

    def evaluate_candidate(self, candidate: Candidate, strategy_name: str = None) -> float:
        """
        evaluate this metric for the candidate, store info in it
        """
        if self.key not in candidate.metrics:
            candidate.metrics[self.key] =\
                self.evaluate_tuple(candidate.values, strategy_name=strategy_name, only_minimize=False)
        return candidate.metrics.get(self.key)

    def evaluate_batch(self, x: np.array, only_minimize=False, strategy_name: str = None) -> np.array:
        """
        evaluate a batch of parameter values at once

        :param x: [batch, ...] numpy array
        :param only_minimize: always return smaller values for better parameters
        :param strategy_name: None if candidate is global, otherwise specific to this weight strategy
        :return: float np.array of how well the batch of given parameter values do
        """
        r = self._evaluate_batch(x, strategy_name=strategy_name)
        if only_minimize:
            return self._sign * r
        return r

    def evaluate_tuple(self, values: tuple, only_minimize=False, strategy_name: str = None) -> float:
        """
        evaluate a single tuple

        :param values: tuple
        :param only_minimize: always return smaller values for better parameters
        :param strategy_name: None if candidate is global, otherwise specific to this weight strategy
        :return: single float value of how well the given parameter values do
        """
        r = self._evaluate_tuple(values, strategy_name=strategy_name)
        if only_minimize:
            return self._sign * r
        return r

    def _evaluate_batch(self, x: np.array, strategy_name: str = None) -> np.array:
        """
        NOTE: either this or the _evaluate_tuple method must be implemented in subclasses
        evaluate a batch of parameter values at once

        :param x: [batch, ...] numpy array
        :param strategy_name: None if candidate is global, otherwise specific to this weight strategy
        :return: float np.array of how well the batch of given parameter values do
        """
        r = [self._evaluate_tuple(xi, strategy_name=strategy_name) for xi in x]
        return np.array(r)

    def _evaluate_tuple(self, values: tuple, strategy_name: str = None) -> float:
        """
        NOTE: either this or the _evaluate_batch method must be implemented in subclasses
        evaluate a single tuple

        :param values: tuple
        :param strategy_name: None if candidate is global, otherwise specific to this weight strategy
        :return: single float value of how well the given parameter values do
        """
        r = self._evaluate_batch(np.array([values], dtype=np.float32), strategy_name=strategy_name)
        return r[0]
