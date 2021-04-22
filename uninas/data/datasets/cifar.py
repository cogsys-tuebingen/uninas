from torchvision import datasets, transforms
from uninas.data.abstract import AbstractCNNClassificationDataSet
from uninas.utils.shape import Shape
from uninas.register import Register


@Register.data_set(images=True, classification=True)
class Cifar10Data(AbstractCNNClassificationDataSet):
    """
    The popular CIFAR-10 data set
    https://www.cs.toronto.edu/~kriz/cifar.html
    """

    length = (50000, 0, 10000)  # training, valid, test
    raw_data_shape = Shape([3, 32, 32])  # channel height width
    raw_label_shape = Shape([10])
    data_mean = (0.49139968, 0.48215827, 0.44653124)
    data_std = (0.24703233, 0.24348505, 0.26158768)

    def _get_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.CIFAR10(root=self.dir, train=True, download=self.download,
                                transform=data_transforms, target_transform=label_transforms)

    def _get_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.CIFAR10(root=self.dir, train=False, download=self.download,
                                transform=data_transforms, target_transform=label_transforms)


@Register.data_set(images=True, classification=True)
class Cifar100Data(AbstractCNNClassificationDataSet):
    """
    The popular CIFAR-100 data set
    https://www.cs.toronto.edu/~kriz/cifar.html
    """

    length = (50000, 0, 10000)  # training, valid, test
    raw_data_shape = Shape([3, 32, 32])  # channel height width
    raw_label_shape = Shape([100])
    data_mean = (0.5071, 0.4867, 0.4408)
    data_std = (0.2675, 0.2565, 0.2761)

    def _get_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.CIFAR100(root=self.dir, train=True, download=self.download,
                                 transform=data_transforms, target_transform=label_transforms)

    def _get_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.CIFAR100(root=self.dir, train=False, download=self.download,
                                 transform=data_transforms, target_transform=label_transforms)
