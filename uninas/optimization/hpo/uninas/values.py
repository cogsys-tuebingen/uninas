import random
import numpy as np
from typing import List
from copy import deepcopy
from collections.abc import Iterable


class AbstractValues:
    """ Values for the hyper-param / gene """

    def get_min_value(self) -> float:
        raise NotImplementedError

    def get_max_value(self) -> float:
        raise NotImplementedError

    def is_allowed(self, v) -> bool:
        raise NotImplementedError

    def sample(self, prev=None):
        raise NotImplementedError


class DiscreteValues(AbstractValues):
    """ Discrete values for the hyper-param / gene """

    def __init__(self, allowed_values: [int], prevent_resample=False):
        self.allowed_values = allowed_values.copy()
        self.prevent_resample = prevent_resample
        assert not (len(allowed_values) <= 1 and prevent_resample)

    @property
    def size(self) -> int:
        return len(self.allowed_values)

    def get_min_value(self) -> int:
        return min(self.allowed_values)

    def get_max_value(self) -> int:
        return max(self.allowed_values)

    def is_allowed(self, v: int) -> bool:
        return v in self.allowed_values

    def sample(self, prev: int = None) -> int:
        while True:
            r = random.randint(0, len(self.allowed_values)-1)
            if r == prev and self.prevent_resample and len(self.allowed_values) > 1:
                continue
            return self.allowed_values[r]

    def remove_value(self, v: int):
        if v in self.allowed_values:
            self.allowed_values.remove(v)

    def as_one_hot(self, v: int, dtype=np.int32) -> np.array:
        arr = np.zeros((self.get_max_value() + 1,), dtype=dtype)
        arr[int(v)] = 1
        return arr

    @classmethod
    def interval(cls, min_val: int, max_val: int) -> AbstractValues:
        """
        :param min_val: inclusive
        :param max_val: exclusive
        """
        return cls(allowed_values=list(range(min_val, max_val)))


class ValueSpace:
    """
    combine AbstractValues
    """

    def __init__(self, *values: [AbstractValues]):
        self.values = values

    def get_values(self) -> [AbstractValues]:
        return self.values

    def num_choices(self) -> int:
        return len(self.values)

    def copy(self):
        return deepcopy(self)

    def __iter__(self):
        for v in self.values:
            yield v

    def is_discrete(self) -> bool:
        return all([isinstance(v, DiscreteValues) for v in self.values])

    def is_allowed(self, values: tuple) -> bool:
        return all([v.is_allowed(vx) or vx == -1 for v, vx in zip(self.values, values)])

    def random_sample(self) -> tuple:
        return tuple([v.sample() for v in self.values])

    def remove_value(self, v: int):
        for value in self.values:
            if isinstance(value, DiscreteValues):
                value.remove_value(v)

    def as_one_hot(self, values: List, dtype=np.int32) -> np.array:
        assert len(values) == self.num_choices(), "mismatching number of values to be represented as one-hot"
        return np.concatenate([val.as_one_hot(v, dtype=dtype) for val, v in zip(self.values, values)], axis=0)

    def iterate(self, fixed_values=None) -> Iterable:
        """ iterate the entire discrete search space, returning tuples """
        assert self.is_discrete()
        if fixed_values is None:
            fixed_values = [-1 for _ in self.values]
        assert len(fixed_values) == len(self.values),\
            "length of fix-description (%d) must match number of values (%d)" % (len(fixed_values), len(self.values))

        def rec(empty: list, fixed: list, depth=0) -> Iterable:
            if depth >= len(self.values):
                yield empty
            elif fixed[depth] >= 0:
                assert fixed[depth] in self.values[depth].allowed_values,\
                    "fixed value %d not allowed at position %d" % (fixed[depth], depth)
                for lst in rec(empty, fixed, depth=depth+1):
                    h = lst.copy()
                    h[depth] = fixed[depth]
                    yield h
            else:
                for v in self.values[depth].allowed_values:
                    for lst in rec(empty, fixed, depth=depth+1):
                        h = lst.copy()
                        h[depth] = v
                        yield h

        for r in rec([0 for _ in self.values], fixed_values):
            yield tuple(r)


class SpecificValueSpace(ValueSpace):
    """
    specific combinations of AbstractValues
    """

    def __init__(self, specific_values: [tuple]):
        super().__init__(None)
        self.specific_values = list(set(specific_values))

    def __len__(self):
        return len(self.specific_values)

    def is_discrete(self) -> bool:
        return isinstance(self.specific_values, (tuple, list))

    def is_allowed(self, values: tuple) -> bool:
        if -1 in values:
            raise NotImplementedError
        return values in self.specific_values

    def random_sample(self) -> tuple:
        return random.choice(self.specific_values)

    def remove_value(self, v: int):
        assert self.is_discrete()
        rem = []
        for i, tpl in enumerate(self.specific_values):
            if v in tpl:
                rem.append(i)
        for i in reversed(rem):
            self.specific_values.pop(i)

    def iterate(self, fixed_values=None) -> Iterable:
        """ iterate the entire discrete search space, returning tuples """
        if fixed_values is not None:
            raise NotImplementedError
        return self.specific_values

    def random_subset(self, k=1) -> 'SpecificValueSpace':
        values = random.sample(self.specific_values, k=k)
        return self.__class__(values)

    @classmethod
    def from_discrete_space(cls, space: ValueSpace):
        values = []
        for v in space.iterate():
            values.append(v)
        return cls(values)


if __name__ == '__main__':
    space_ = ValueSpace(
        DiscreteValues([0, 1, 2]),
        DiscreteValues([1, 7]),
        DiscreteValues.interval(0, 4),
    )
    for a in space_.iterate([-1, -1, 3]):
        print(a, '\t', space_.as_one_hot(a))
    s2 = SpecificValueSpace(space_.iterate())
    print(list(s2.random_subset(k=3).iterate()))
