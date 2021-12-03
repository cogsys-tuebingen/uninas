import os
import numpy as np
import torch
from uninas.utils.args import ArgsInterface, Namespace
from uninas.utils.paths import replace_standard_paths
from uninas.utils.types import CheckpointType
from uninas.utils.np import squeeze_keep_batch
from uninas.register import Register


class AbstractModel(ArgsInterface):

    def save(self, path: str):
        """ save this model to 'path' """
        path = replace_standard_paths(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.get_model_state(), path)

    def get_model_state(self) -> dict:
        """ get a state dict that can later recover the model """
        return {
            '__type__': CheckpointType.MODEL.value,
            '__cls__': self.__class__.__name__,
            'state': self._get_model_state(),
        }

    def load(self, path: str) -> bool:
        """ update the current model with the state saved in 'path' """
        path = replace_standard_paths(path)
        if os.path.isfile(path):
            return self.load_state(torch.load(path))
        return False

    def load_state(self, state: dict) -> bool:
        """ update the current model with this state """
        if '__type__' in state:
            assert CheckpointType.MODEL == state['__type__']
            assert state['__cls__'] == self.__class__.__name__
            state = state['state']
        return self._load_state(state)

    @classmethod
    def load_from(cls, path: str) -> 'AbstractModel':
        """ load the model that's been saved in 'path' """
        path = replace_standard_paths(path)
        state = torch.load(path)
        assert '__cls__' in state, "Missing class info! Expected a model save file, but this is not one: %s" % path
        used_cls = Register.get(state['__cls__'])
        assert issubclass(used_cls, AbstractModel), "%s is not a subclass of %s" % (used_cls.__name__, AbstractModel.__name__)
        return used_cls._load_from(state['state'])

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> 'AbstractModel':
        """
        :param args: global argparse namespace
        :param index: argument index
        """
        raise NotImplementedError

    def _get_model_state(self) -> dict:
        """ get a state dict that can later recover the model """
        raise NotImplementedError

    def _load_state(self, model_state: dict) -> bool:
        """ update the current model with this state """
        raise NotImplementedError

    @classmethod
    def _load_from(cls, model_state: dict) -> 'AbstractModel':
        """ create this model from a state dict """
        raise NotImplementedError

    def prepare_predict(self, device: str) -> 'AbstractModel':
        """ place the model on some hardware device, go eval mode """
        return self

    def fit(self, data: np.array, labels: np.array):
        """
        fit the model to data+labels
        :param data: n-dimensional np array, first dimension is the batch
        :param labels: n-dimensional np array, first dimension is the batch
        :return:
        """
        raise NotImplementedError

    def predict(self, data: np.array) -> np.array:
        """
        predict the labels of the data
        :param data: n-dimensional np array, first dimension is the batch
        :return: n-dimensional np array, first dimension is the batch
        """
        raise NotImplementedError


class AbstractWrapperModel(AbstractModel):
    """
    An abstract model that wraps a model from some external source
    """
    _none_args = []
    _model_cls = None

    def __init__(self, model=None, **kwargs):
        super().__init__()

        # remove args that are None by default but set to -1 for Argument compatibility
        for k in self._none_args:
            v = kwargs.get(k, None)
            if isinstance(v, int) and v < 0:
                del kwargs[k]

        self._model_kwargs = kwargs
        self.model = model if model is not None else self._model_cls(**kwargs)

    @classmethod
    def from_args(cls, args: Namespace, index: int = None) -> 'AbstractWrapperModel':
        parsed = cls._all_parsed_arguments(args, index=index)
        return cls(**parsed)

    def _get_model_state(self) -> dict:
        """ get a state dict that can later recover the model """
        return dict(model=self.model, kwargs=self._model_kwargs)

    def _load_state(self, model_state: dict) -> bool:
        """ update the current model with this state """
        model = model_state.get('model')
        kwargs = model_state.get('kwargs')
        assert model.__class__ == self.model.__class__, "Used another model in the save file"
        assert kwargs == self._model_kwargs, "Used different model params in the model state"
        self.model = model
        self._model_kwargs = kwargs
        return True

    @classmethod
    def _load_from(cls, model_state: dict) -> AbstractModel:
        """ create this model from a state dict """
        return cls(model=model_state.get('model'), **model_state.get('kwargs'))

    def _str_dict(self) -> dict:
        return self._model_kwargs

    def fit(self, data: np.array, labels: np.array):
        """
        fit the model to data+labels
        :param data: n-dimensional np array, first dimension is the batch
        :param labels: n-dimensional np array, first dimension is the batch
        :return:
        """
        self.model.fit(squeeze_keep_batch(data), squeeze_keep_batch(labels))

    def predict(self, data: np.array) -> np.array:
        """
        predict the labels of the data
        :param data: n-dimensional np array, first dimension is the batch
        :return: n-dimensional np array, first dimension is the batch
        """
        return self.model.predict(squeeze_keep_batch(data))
