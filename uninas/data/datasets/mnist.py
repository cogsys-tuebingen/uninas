"""
"""

from torchvision import datasets, transforms
from uninas.data.abstract import AbstractCNNClassificationDataSet
from uninas.utils.shape import Shape
from uninas.utils.args import Namespace
from uninas.register import Register


@Register.data_set(images=True)
class MnistData(AbstractCNNClassificationDataSet):
    length = (60000, 0, 10000)  # training, valid, test
    data_raw_shape = Shape([1, 28, 28])  # channel height width
    label_shape = Shape([10])
    data_mean = (0.1307,)
    data_std = (0.3081,)

    def _get_train_data(self, args: Namespace, used_transforms: transforms.Compose):
        return datasets.MNIST(root=self.dir, train=True, download=self.download, transform=used_transforms)

    def _get_test_data(self, args: Namespace, used_transforms: transforms.Compose):
        return datasets.MNIST(root=self.dir, train=False, download=self.download, transform=used_transforms)


@Register.data_set(images=True)
class FashionMnistData(AbstractCNNClassificationDataSet):
    length = (60000, 0, 10000)  # training, valid, test
    data_raw_shape = Shape([1, 28, 28])  # channel height width
    label_shape = Shape([10])
    data_mean = (0.2860,)
    data_std = (0.3530,)

    def _get_train_data(self, args: Namespace, used_transforms: transforms.Compose):
        return datasets.FashionMNIST(root=self.dir, train=True, download=self.download, transform=used_transforms)

    def _get_test_data(self, args: Namespace, used_transforms: transforms.Compose):
        return datasets.FashionMNIST(root=self.dir, train=False, download=self.download, transform=used_transforms)
