import unittest
from uninas.models.abstract import AbstractModel
from uninas.utils.torch.standalone import get_dataset
from uninas.utils.loggers.python import LoggerManager
from uninas.register import Register, RegisteredItem
from uninas.builder import Builder


class TestModels(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = LoggerManager().get_logger()

    def test_reg_model_save_load(self):
        """
        expected output shapes of standard layers
        """
        Builder()
        save_path = "{path_tmp}/tests/model.pt"
        data_set = get_dataset({
            "cls_data": "ProfiledData",
            "ProfiledData.valid_split": 0.1,
            "ProfiledData.dir": "{path_data}/profiling/HW-NAS/",
            "ProfiledData.file_name": "ImageNet16-120-raspi4_latency.pt",
            "ProfiledData.cast_one_hot": True,
            "ProfiledData.train_num": 100,

            "cls_augmentations": "",
        })
        data_train = data_set.get_full_train_data(to_numpy=True, num=100)
        data_test = data_set.get_full_test_data(to_numpy=True, num=100)

        for item in Register.models.filter_match_all(can_fit=True, regression=True).values():
            assert isinstance(item, RegisteredItem)
            model_cls = item.value
            assert isinstance(model_cls, type)
            assert issubclass(model_cls, AbstractModel)
            model_kwargs = model_cls.parsed_argument_defaults()

            self.logger.info("creating %s with default argements: %s" % (model_cls.__name__, repr(model_kwargs)))
            model = model_cls(**model_kwargs)

            # fit, get results, save
            model.fit(*data_train)
            results = model.predict(data_test[0])
            model.save(save_path)
            self.logger.info(' -> fit and saved')

            # load other model, get results
            model2 = AbstractModel.load_from(save_path)
            results2 = model2.predict(data_test[0])
            self.logger.info(' -> loaded')

            # compare
            assert model.__class__ == model2.__class__,\
                "Classes of models do not match: (%s, %s)" % (model.__class__, model2.__class__)
            for v1, v2 in zip(results, results2):
                assert v1 - 0.0001 < v2 < v1 + 0.0001
            self.logger.info(' -> same results for both, save+load works')


if __name__ == '__main__':
    unittest.main()
