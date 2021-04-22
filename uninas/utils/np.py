from typing import Iterable
import numpy as np


def squeeze_keep_batch(v: np.array) -> np.array:
    """ squeeze, except for the batch dimension """
    expand_back = v.shape[0] == 1
    v = np.squeeze(v)
    if expand_back:
        return np.expand_dims(v, axis=0)
    return v


def concatenated_one_hot(values: Iterable, max_sizes: Iterable, dtype=np.float32) -> np.array:
    oh_values = []
    for v, m in zip(values, max_sizes):
        o = np.zeros(shape=(m,), dtype=dtype)
        o[v] = 1
        oh_values.append(o)
    return np.concatenate(oh_values, axis=0)
