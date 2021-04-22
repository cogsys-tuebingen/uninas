import os
from torchvision import datasets, transforms
from uninas.data.abstract import AbstractCNNClassificationDataSet
from uninas.utils.shape import Shape
from uninas.register import Register


@Register.data_set(images=True, classification=True)
class Cinic10Data(AbstractCNNClassificationDataSet):
    """
    CINIC-10: CINIC-10 Is Not Imagenet or CIFAR-10
    https://github.com/BayesWatch/cinic-10
    """

    length = (90000, 90000, 90000)  # training, valid, test
    raw_data_shape = Shape([3, 32, 32])
    raw_label_shape = Shape([10])
    data_mean = (0.47889522, 0.47227842, 0.43047404)
    data_std = (0.24205776, 0.23828046, 0.25874835)

    def _get_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.ImageFolder(os.path.join(self.dir, 'train'),
                                    transform=data_transforms, target_transform=label_transforms)

    def _get_valid_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.ImageFolder(os.path.join(self.dir, 'valid'),
                                    transform=data_transforms, target_transform=label_transforms)

    def _get_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.ImageFolder(os.path.join(self.dir, 'test'),
                                    transform=data_transforms, target_transform=label_transforms)
