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

    length = (1281167, 50000, 0)  # training, valid, test
    raw_data_shape = Shape([3, 300, 300])  # channel height width, the shapes of the raw images actually vary
    raw_label_shape = Shape([1000])
    data_mean = (0.485, 0.456, 0.406)
    data_std = (0.229, 0.224, 0.225)

    can_download = False

    def _get_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.ImageFolder(os.path.join(self.dir, 'train'),
                                    transform=data_transforms, target_transform=label_transforms)

    def _get_valid_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return datasets.ImageFolder(os.path.join(self.dir, 'val'),
                                    transform=data_transforms, target_transform=label_transforms)
