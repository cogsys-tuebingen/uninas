import unittest
import numpy as np
from typing import Callable
from uninas.data.abstract import AbstractDataSet
from uninas.models.abstract import AbstractModel
from uninas.models.networks.mini.fully_connected import FullyConnectedNetwork
from uninas.models.networks.abstract import AbstractNetwork
from uninas.utils.torch.standalone import get_dataset
from uninas.utils.loggers.python import LoggerManager
from uninas.register import Register, RegisteredItem
from uninas.builder import Builder


class TestModels(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Builder()
        self.logger = LoggerManager().get_logger()
        self.save_path = "{path_tmp}/tests/model.pt"

    @classmethod
    def _get_reg_data(cls) -> AbstractDataSet:
        return get_dataset({
            "cls_data": "ProfiledData",
            "ProfiledData.valid_split": 0.1,
            "ProfiledData.dir": "{path_data}/profiling/HW-NAS/",
            "ProfiledData.file_name": "ImageNet16-120-raspi4_latency.pt",
            "ProfiledData.cast_one_hot": True,
            "ProfiledData.train_num": 100,

            "cls_augmentations": "",
        })

    def _create_default_model(self, model_cls: Callable) -> AbstractModel:
        assert isinstance(model_cls, type)
        assert issubclass(model_cls, AbstractModel)
        model_kwargs = model_cls.parsed_argument_defaults()
        assert isinstance(model_kwargs, dict)
        self.logger.info("creating %s with default arguments: %s" % (model_cls.__name__, repr(model_kwargs)))
        return model_cls(**model_kwargs)

    def _save_load_assert(self, model: AbstractModel, data: np.array):
        # save model
        model.save(self.save_path)
        self.logger.info(' -> saved')

        # load other model, get results
        model2 = AbstractModel.load_from(self.save_path)
        self.logger.info(' -> loaded anonymously')

        # predict on both
        results = model.predict(data)
        results2 = model2.predict(data)
        self.logger.info(' -> predicted both')

        # compare
        assert model.__class__ == model2.__class__,\
            "Classes of models do not match: (%s, %s)" % (model.__class__, model2.__class__)
        for v1, v2 in zip(results, results2):
            assert v1 - 0.0001 < v2 < v1 + 0.0001
        self.logger.info(' -> same results, save+load works')

    def test_nn_save_load(self):
        """
        create simple networks, save and load, make sure that the predictions are the same
        """
        data_set = self._get_reg_data()
        data_test = data_set.get_full_test_data(to_numpy=True, num=100)

        # get network, build, randomize to simulate training
        model = self._create_default_model(FullyConnectedNetwork)
        assert isinstance(model, AbstractNetwork)
        model.build(data_set.get_data_shape(), data_set.get_label_shape())
        model.randomize_parameters()
        self.logger.info(' -> built and randomized')

        self._save_load_assert(model, data_test[0])

    def test_reg_model_save_load(self):
        """
        fit regression models, save and load, make sure that the predictions are the same
        """
        data_set = self._get_reg_data()
        data_train = data_set.get_full_train_data(to_numpy=True, num=100)
        data_test = data_set.get_full_test_data(to_numpy=True, num=100)

        for item in Register.models.filter_match_all(can_fit=True, regression=True).values():
            assert isinstance(item, RegisteredItem)
            model_cls = item.value

            # get regression model, fit
            model = self._create_default_model(model_cls)
            model.fit(*data_train)
            self.logger.info(' -> fit')

            self._save_load_assert(model, data_test[0])


if __name__ == '__main__':
    unittest.main()
