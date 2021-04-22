import inspect
from collections.abc import Callable
from typing import Union


def split(string: str, cast_fun: Callable = None) -> list:
    """ split a comma-separated string into a list, optionally cast list elements to desired type """
    if string is None or len(string) == 0:
        return []
    re = list(string.replace(' ', '').split(','))
    if (cast_fun is not None) and (not isinstance(cast_fun, bool)):
        re = [cast_fun(s) for s in re]
    return re


def flatten(nested: [[]]) -> []:
    """ flatten a nested list """
    flattened = []
    for n in nested:
        if isinstance(n, list):
            flattened.extend(flatten(n))
        else:
            flattened.append(n)
    return flattened


def add_to_dict_key(prefix: str, dct: dict) -> dict:
    """ add a prefix to every dict key """
    return {'%s/%s' % (prefix, k): v for k, v in dct.items()}


def get_number(number: Union[int, float], default: Union[int, float]):
    """ get default if the number is None or <0, else the number """
    if number is None or number < 0:
        return default
    return number


def get_var_name(variable):
    """ get the name of a variable """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is variable]
        if len(names) > 0:
            return names[0]


def power_list(lists: [list]) -> list:
    """ power set across the options of all lists """
    if len(lists) == 1:
        return [[v] for v in lists[0]]
    grids = power_list(lists[:-1])
    new_grids = []
    for v in lists[-1]:
        for g in grids:
            new_grids.append(g + [v])
    return new_grids


if __name__ == "__main__":

    a1 = [1, 2, 3]
    b1 = ['a', 'b']
    c1 = [10, 11]

    for s1 in power_list([b1, a1, c1]):
        print(s1)
