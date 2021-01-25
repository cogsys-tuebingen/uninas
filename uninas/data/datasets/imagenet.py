import os
from torchvision import datasets, transforms
from uninas.data.abstract import AbstractCNNClassificationDataSet
from uninas.utils.shape import Shape
from uninas.register import Register


@Register.data_set(images=True, classification=True)
class Imagenet1000Data(AbstractCNNClassificationDataSet):
    """
    The ImageNet data set
    http://image-net.org/
    """

    length = (1281167, 0, 50000)  # training, valid, test
    data_raw_shape = Shape([3, 300, 300])  # channel height width, the shapes of the raw images actually vary
    label_shape = Shape([1000])
    data_mean = (0.485, 0.456, 0.406)
    data_std = (0.229, 0.224, 0.225)

    can_download = False

    def _get_train_data(self, used_transforms: transforms.Compose):
        return datasets.ImageFolder(os.path.join(self.dir, 'train'), transform=used_transforms)

    def _get_test_data(self, used_transforms: transforms.Compose):
        return datasets.ImageFolder(os.path.join(self.dir, 'val'), transform=used_transforms)


@Register.data_set(images=True, classification=True)
class SubImagenet100Data(Imagenet1000Data):
    """
    Subset of the ImageNet data set with fewer classes, and fewer images per class
    http://image-net.org/
    https://github.com/microsoft/Cream/blob/main/tools/generate_subImageNet.py
    """

    length = (25000, 0, 5000)  # training, valid, test
    data_raw_shape = Shape([3, 300, 300])  # channel height width, the shapes of the raw images actually vary
    label_shape = Shape([100])
    data_mean = (0.485, 0.456, 0.406)  # not recomputed for the subset
    data_std = (0.229, 0.224, 0.225)   # not recomputed for the subset

    can_download = False
