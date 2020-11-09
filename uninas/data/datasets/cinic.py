import os
from torchvision import datasets, transforms
from uninas.data.abstract import AbstractCNNClassificationDataSet
from uninas.utils.shape import Shape
from uninas.utils.args import Namespace
from uninas.register import Register


@Register.data_set(images=True)
class Cinic10Data(AbstractCNNClassificationDataSet):
    """
    CINIC-10: CINIC-10 Is Not Imagenet or CIFAR-10
    https://github.com/BayesWatch/cinic-10
    """

    length = (90000, 90000, 90000)  # training, valid, test
    data_raw_shape = Shape([3, 32, 32])
    label_shape = Shape([10])
    data_mean = (0.47889522, 0.47227842, 0.43047404)
    data_std = (0.24205776, 0.23828046, 0.25874835)

    def _get_train_data(self, args: Namespace, used_transforms: transforms.Compose):
        return datasets.ImageFolder(os.path.join(self.dir, 'train'), transform=used_transforms)

    def _get_valid_data(self, args: Namespace, used_transforms: transforms.Compose):
        return datasets.ImageFolder(os.path.join(self.dir, 'valid'), transform=used_transforms)

    def _get_test_data(self, args: Namespace, used_transforms: transforms.Compose):
        return datasets.ImageFolder(os.path.join(self.dir, 'test'), transform=used_transforms)
