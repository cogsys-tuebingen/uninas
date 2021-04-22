"""
"""

from torchvision import datasets, transforms
from uninas.data.abstract import AbstractCNNClassificationDataSet
from uninas.utils.shape import Shape
from uninas.register import Register


@Register.data_set(images=True, classification=True)
class MnistData(AbstractCNNClassificationDataSet):
    """
    Handwritten digits from 0 to 9
    """

    length = (60000, 0, 10000)  # training, valid, test
    raw_data_shape = Shape([1, 28, 28])  # channel height width
    raw_label_shape = Shape([10])
    data_mean = (0.1307,)
    data_std = (0.3081,)

    @classmethod
    def get_class_names(cls) -> [str]:
        return datasets.MNIST.classes

    def _get_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.MNIST(root=self.dir, train=True, download=self.download,
                              transform=data_transforms, target_transform=label_transforms)

    def _get_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.MNIST(root=self.dir, train=False, download=self.download,
                              transform=data_transforms, target_transform=label_transforms)


@Register.data_set(images=True, classification=True)
class FashionMnistData(AbstractCNNClassificationDataSet):
    """
    """

    length = (60000, 0, 10000)  # training, valid, test
    raw_data_shape = Shape([1, 28, 28])  # channel height width
    raw_label_shape = Shape([10])
    data_mean = (0.2860,)
    data_std = (0.3530,)

    @classmethod
    def get_class_names(cls) -> [str]:
        return datasets.FashionMNIST.classes

    def _get_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.FashionMNIST(root=self.dir, train=True, download=self.download,
                                     transform=data_transforms, target_transform=label_transforms)

    def _get_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.FashionMNIST(root=self.dir, train=False, download=self.download,
                                     transform=data_transforms, target_transform=label_transforms)
