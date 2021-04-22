"""
downsampled ImageNet
the code below is mostly original

original author and copyright: Xuanyi Dong [GitHub D-X-Y], 2019
see https://github.com/D-X-Y/AutoDL-Projects/blob/ec4f9c40c7e1e7afdc07fc58e23f3006435edc76/lib/datasets/DownsampledImageNet.py
"""

import os
import sys
import hashlib
import pickle
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from uninas.data.abstract import AbstractCNNClassificationDataSet
from uninas.utils.shape import Shape
from uninas.utils.paths import replace_standard_paths
from uninas.builder import Builder
from uninas.register import Register


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath): return False
    if md5 is None:
        return True
    else:
        return check_md5(fpath, md5)


class ImageNet16(Dataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    train_list = [
        ['train_data_batch_1', '27846dcaa50de8e21a7d1a35f30f0e91'],
        ['train_data_batch_2', 'c7254a054e0e795c69120a5727050e3f'],
        ['train_data_batch_3', '4333d3df2e5ffb114b05d2ffc19b1e87'],
        ['train_data_batch_4', '1620cdf193304f4a92677b695d70d10f'],
        ['train_data_batch_5', '348b3c2fdbb3940c4e9e834affd3b18d'],
        ['train_data_batch_6', '6e765307c242a1b3d7d5ef9139b48945'],
        ['train_data_batch_7', '564926d8cbf8fc4818ba23d2faac7564'],
        ['train_data_batch_8', 'f4755871f718ccb653440b9dd0ebac66'],
        ['train_data_batch_9', 'bb6dd660c38c58552125b1a92f86b5d4'],
        ['train_data_batch_10', '8f03f34ac4b42271a294f91bf480f29b'],
    ]
    valid_list = [
        ['val_data', '3410e3017fdaefba8d5073aaa65e4bd6'],
    ]

    def __init__(self, root, train, transform, label_transform, use_num_of_class_only=None):
        self.root = root
        self.transform = transform
        self.label_transform = label_transform
        self.train = train  # training set or valid set
        if not self._check_integrity(): raise RuntimeError('Dataset not found or corrupted.')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for i, (file_name, checksum) in enumerate(downloaded_list):
            file_path = os.path.join(self.root, file_name)
            # print ('Load {:}/{:02d}-th : {:}'.format(i, len(downloaded_list), file_path))
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if use_num_of_class_only is not None:
            assert isinstance(use_num_of_class_only, int)\
                   and (use_num_of_class_only > 0)\
                   and (use_num_of_class_only < 1000),\
                'invalid use_num_of_class_only: {:}'.format(use_num_of_class_only)
            new_data, new_targets = [], []
            for I, L in zip(self.data, self.targets):
                if 1 <= L <= use_num_of_class_only:
                    new_data.append(I)
                    new_targets.append(L)
            self.data = new_data
            self.targets = new_targets
        #    self.mean.append(entry['mean'])
        # self.mean = np.vstack(self.mean).reshape(-1, 3, 16, 16)
        # self.mean = np.mean(np.mean(np.mean(self.mean, axis=0), axis=1), axis=1)
        # print ('Mean : {:}'.format(self.mean))
        # temp      = self.data - np.reshape(self.mean, (1, 1, 1, 3))
        # std_data  = np.std(temp, axis=0)
        # std_data  = np.mean(np.mean(std_data, axis=0), axis=0)
        # print ('Std  : {:}'.format(std_data))

    def __repr__(self):
        return ('{name}({num} images, {classes} classes)'.format(name=self.__class__.__name__, num=len(self.data),
                                                                 classes=len(set(self.targets))))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index] - 1

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            target = self.label_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.valid_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


@Register.data_set(images=True, classification=True)
class ImageNet16Data(AbstractCNNClassificationDataSet):
    """
    A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets

    https://arxiv.org/pdf/1707.08819.pdf
    http://image-net.org/download-images
    https://github.com/PatrykChrabaszcz/Imagenet32_Scripts
    """

    length = (1281167, 0, 50000)  # training, valid, test
    raw_data_shape = Shape([3, 16, 16])  # channel height width
    raw_label_shape = Shape([1000])
    data_mean = (0.485, 0.456, 0.406)   # original imagenet values
    data_std = (0.229, 0.224, 0.225)    # original imagenet values

    can_download = False

    def _num_classes(self):
        nc = self.get_label_shape().num_features()
        if nc >= 1000:
            return None
        return nc

    def _get_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return ImageNet16(root=self.dir, train=True, transform=data_transforms, label_transform=label_transforms,
                          use_num_of_class_only=self._num_classes())

    def _get_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return ImageNet16(root=self.dir, train=False, transform=data_transforms, label_transform=label_transforms,
                          use_num_of_class_only=self._num_classes())


@Register.data_set(images=True, classification=True)
class ImageNet16c120Data(ImageNet16Data):
    """
    A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets,
    reduced to 120/1000 classes

    https://arxiv.org/pdf/1707.08819.pdf
    http://image-net.org/download-images
    https://github.com/PatrykChrabaszcz/Imagenet32_Scripts
    """

    length = (151700, 0, 6000)  # training, valid, test
    raw_label_shape = Shape([120])


if __name__ == '__main__':
    Builder()
    path = replace_standard_paths('{path_data}/ImageNet16')
    train = ImageNet16(path, True, None, use_num_of_class_only=120)
    valid = ImageNet16(path, False, None, use_num_of_class_only=120)
    print(len(train))
    print(len(valid))
    img = train[14]
    print(img)
    print(max(train.targets))
    print(max(valid.targets))
