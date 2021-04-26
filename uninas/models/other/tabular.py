"""
models from scikit-learn
"""

import numpy as np
from uninas.models.abstract import AbstractModel
from uninas.utils.paths import replace_standard_paths
from uninas.register import Register
from uninas.builder import Builder


@Register.model(regression=True)
class TabularSumModel(AbstractModel):
    """
    A lookup model that sums up values from a table, depending on the query

    the table has the form:
    table[cell index][op index] = value
    """

    def __init__(self, table: dict, constant: float = 0.0):
        super().__init__()
        self.table = table
        self.constant = constant

    @classmethod
    def from_args(cls, *_, **__) -> 'AbstractModel':
        return cls(table={}, constant=0.0)

    def _get_model_state(self) -> dict:
        """ get a state dict that can later recover the model """
        return dict(table=self.table, constant=self.constant)

    def _load_state(self, model_state: dict) -> bool:
        """ update the current model with this state """
        self.table = model_state.get('table')
        self.constant = model_state.get('constant')
        return True

    @classmethod
    def _load_from(cls, model_state: dict) -> AbstractModel:
        """ create this model from a state dict """
        return cls(table=model_state.get('table'), constant=model_state.get('constant'))

    def fit(self, data: np.array, labels: np.array):
        """
        fit the model to data+labels
        :param data: n-dimensional np array, first dimension is the batch
        :param labels: n-dimensional np array, first dimension is the batch
        :return:
        """
        raise NotImplementedError("This model type can not be fit to data, it uses a fixed lookup table")

    def predict(self, data: np.array) -> np.array:
        """
        predict the labels of the data
        sum up values for a batch of: [op idx in cell 0, ... op idx in cell -1]
        requires 2D data

        :param data: n-dimensional np array, first dimension is the batch
        :return: n-dimensional np array, first dimension is the batch
        """
        assert len(data.shape) == 2, "Can only look up 2D (batch, indices) data in the tables"
        values = np.zeros(shape=(len(data),))
        values += self.constant
        for i, arr in enumerate(data):
            for j, a in enumerate(arr):
                values[i] += self.table[j][a]
        return values


if __name__ == '__main__':
    Builder()
    model_ = TabularSumModel.load_from(replace_standard_paths("{path_profiled}/HW-NAS/tab-fbnet-cifar100-edgegpu_latency.pt"))
    arcs = [
        [0]*22,
        [1]*22,
        [2]*22,
    ]
    print(model_.predict(np.array(arcs)))
