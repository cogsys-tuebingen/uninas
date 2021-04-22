"""
random data
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from uninas.data.abstract import AbstractDataSet
from uninas.utils.misc import split
from uninas.utils.shape import Shape
from uninas.utils.args import Namespace, Argument
from uninas.register import Register


class RandomDataset(Dataset):
    def __init__(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose,
                 is_image: bool, num=1000, data_size=(10, 10, 10), target_size=(1,), classification=True):
        self.data_transforms = data_transforms
        self.is_image = is_image
        self.num = num
        self.data_size = data_size
        self.target_size = target_size
        self.classification = classification

        # maybe change the indices
        if self.is_image:
            assert len(data_size) == 3
            self.data_size = [self.data_size[1], self.data_size[2], self.data_size[0]]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.is_image:
            x = Image.fromarray(np.random.randint(0, 255, dtype=np.uint8, size=self.data_size))
        else:
            x = np.random.normal(size=self.data_size).astype(np.float32)
        data = self.data_transforms(x)
        if self.classification:
            return data, np.random.randint(low=0, high=self.target_size[0], size=[1])
        return data, np.random.normal(size=self.target_size).astype(np.float32)


class RandomData(AbstractDataSet):
    length = (10000, 0, 5000)  # training, valid, test
    raw_data_shape = None   # set via args
    raw_label_shape = None      # set via args
    data_mean = (0.0,)
    data_std = (1.0,)

    @classmethod
    def from_args(cls, args: Namespace, index: int = None) -> AbstractDataSet:
        # set class attributes
        data_shape, target_shape = cls._parsed_arguments(['data_shape', 'target_shape'], args, index=index)
        cls.raw_data_shape = Shape(split(data_shape, int))
        cls.raw_label_shape = Shape(split(target_shape, int))

        # default generation now
        return super().from_args(args, index)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        args = super().args_to_add(index) + [
            Argument('data_shape', default='3, 224, 224', type=str, help='shape of the data'),
            Argument('target_shape', default='1000', type=str, help='shape of the targets'),
        ]
        return args

    def _get_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return RandomDataset(data_transforms, label_transforms, is_image=self.is_on_images(),
                             num=self.length[0], data_size=self.raw_data_shape.shape,
                             target_size=self.label_shape.shape, classification=self.is_classification())

    def _get_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return RandomDataset(data_transforms, label_transforms, is_image=self.is_on_images(),
                             num=self.length[2], data_size=self.raw_data_shape.shape,
                             target_size=self.label_shape.shape, classification=self.is_classification())

    def _get_fake_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return self._get_train_data(data_transforms, label_transforms)

    def _get_fake_valid_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return self._get_valid_data(data_transforms, label_transforms)

    def _get_fake_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return self._get_test_data(data_transforms, label_transforms)


@Register.data_set(classification=True, images=True)
class RandomImageClassificationData(RandomData):
    """
    random data and targets, emulating image classification
    """


@Register.data_set(regression=True)
class RandomRegressionData(RandomData):
    """
    random data and targets, emulating regression
    """
