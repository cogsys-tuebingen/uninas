from uninas.data.datasets.imagenet import Imagenet1000Data
from uninas.utils.shape import Shape
from uninas.register import Register


@Register.data_set(images=True, classification=True)
class SubImagenet100Data(Imagenet1000Data):
    """
    Subset of the ImageNet data set with fewer classes, and fewer images per class
    http://image-net.org/
    https://github.com/microsoft/Cream/blob/main/tools/generate_subImageNet.py

    A script to generate this dataset from ImageNet is in the utils
    """

    length = (25000, 5000, 0)  # training, valid, test
    raw_data_shape = Shape([3, 300, 300])  # channel height width, the shapes of the raw images actually vary
    raw_label_shape = Shape([100])
    data_mean = (0.485, 0.456, 0.406)  # not recomputed for the subset
    data_std = (0.229, 0.224, 0.225)   # not recomputed for the subset

    can_download = False


@Register.data_set(images=True, classification=True)
class SubImagenetMV100Data(SubImagenet100Data):
    """
    Subset of the ImageNet data set with fewer classes, and fewer images per class
    http://image-net.org/
    https://github.com/microsoft/Cream/blob/main/tools/generate_subImageNet.py
    This data set variation simply contains many more validation images than the original by Microsoft

    A script to generate this dataset from ImageNet is in the utils
    """

    length = (25000, 25000, 0)  # training, valid, test


@Register.data_set(images=True, classification=True)
class SubImagenetc100t1000v500Data(SubImagenet100Data):
    """
    Subset of the ImageNet data set with fewer classes, and fewer images per class
    http://image-net.org/
    https://github.com/microsoft/Cream/blob/main/tools/generate_subImageNet.py
    This data set variation simply contains many more validation images than the original by Microsoft

    A script to generate this dataset from ImageNet is in the utils
    """

    length = (100000, 50000, 0)  # training, valid, test
