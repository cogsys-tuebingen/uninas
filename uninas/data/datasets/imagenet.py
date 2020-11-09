"""
http://image-net.org/
"""

import os
from torchvision import datasets, transforms
from uninas.data.abstract import AbstractCNNClassificationDataSet
from uninas.utils.shape import Shape
from uninas.utils.args import Namespace
from uninas.register import Register


@Register.data_set(images=True)
class Imagenet1000Data(AbstractCNNClassificationDataSet):
    length = (1281167, 0, 50000)  # training, valid, test
    data_raw_shape = Shape([3, 300, 300])  # channel height width, the shapes of the raw images actually vary
    label_shape = Shape([1000])
    data_mean = (0.485, 0.456, 0.406)
    data_std = (0.229, 0.224, 0.225)

    def _get_train_data(self, args: Namespace, used_transforms: transforms.Compose):
        return datasets.ImageFolder(os.path.join(self.dir, 'train'), transform=used_transforms)

    def _get_test_data(self, args: Namespace, used_transforms: transforms.Compose):
        return datasets.ImageFolder(os.path.join(self.dir, 'val'), transform=used_transforms)
