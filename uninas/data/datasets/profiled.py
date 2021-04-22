"""
profiled data
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from uninas.data.abstract import AbstractDataSet
from uninas.utils.shape import Shape
from uninas.utils.args import Argument
from uninas.utils.np import concatenated_one_hot
from uninas.register import Register


class VectorDataset(Dataset):
    def __init__(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose,
                 path: str, use='train', cast_one_hot=True, scale=1, num=-1):
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms
        self.scale = scale
        all_data = torch.load(path)
        sizes = all_data['sizes']
        data = all_data[use]
        keys = list(data.keys())
        if num > 0:
            keys = keys[:num]
        self.keys = [concatenated_one_hot(k, sizes, dtype=np.float32) for k in keys] if cast_one_hot else\
            [np.array(k, dtype=np.float32) for k in keys]
        self.values = [np.array([data.get(k) * scale], dtype=np.float32) for k in keys]

    def __len__(self):
        return len(self.keys)

    def get_input_len(self) -> int:
        key, _ = self[0]
        return len(key)

    def __getitem__(self, idx):
        return self.data_transforms(self.keys[idx]), self.label_transforms(self.values[idx])


@Register.data_set(regression=True)
class ProfiledData(AbstractDataSet):
    """
    Dataset for fitting profiling predictors

    The pytorch save file is expected to contain a dict
    {
        sizes: tuple,
        train: {tuple: float},
        test: {tuple: float}
    }
    where sizes contains the highest possible index per architecture choice (thus enables one-hot encoding),
    and the train/test tuples contain index tuples of the architecture and the profiled value (e.g. {(2, 1, 3): 0.1})
    """

    length = [-1, 0, -1]  # training, valid, test; depends on the loaded data
    raw_data_shape = Shape([10])    # depends on the loaded data
    raw_label_shape = Shape([1])
    data_mean = (0.0,)
    data_std = (1.0,)

    @classmethod
    def separate_and_save(cls, path_save: str, data: dict, sizes: tuple, num_test=1000, shuffle=False):
        """
        separate profiled data into a training and test set, save it
        :param path_save: where to save to
        :param data: dict of {tuple: float}
        :param sizes: max index for each choice
        :param num_test: num data points to reserve for the test set
        :param shuffle: whether to shuffle data
        :return:
        """
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        keys = list(data.keys())
        if shuffle:
            random.shuffle(keys)
        data_train = {k: data.get(k) for k in keys[:-num_test]}
        data_test = {k: data.get(k) for k in keys[-num_test:]}
        torch.save(dict(sizes=sizes, train=data_train, test=data_test), path_save)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('file_name', default="profiled.pt", type=str, help='name of the save file'),
            Argument('train_num', default=-1, type=int, help='reduce the training set to this number of data points'),
            Argument('cast_one_hot', default="True", type=str, help='cast the training inputs to one-hot', is_bool=True),
            Argument('scale', default=1.0, type=float, help='scale the output'),
        ]

    def _get_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        fn, scale = self.additional_args.get('file_name'), self.additional_args.get('scale')
        cast_one_hot, num = self.additional_args.get('cast_one_hot'), self.additional_args.get('train_num')
        ds = VectorDataset(data_transforms, label_transforms, path="%s/%s" % (self.dir, fn), use="train",
                           cast_one_hot=cast_one_hot, scale=scale, num=num)
        self.length[0] = len(ds)
        self.raw_data_shape = Shape([ds.get_input_len()])
        return ds

    def _get_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        fn, scale = self.additional_args.get('file_name'), self.additional_args.get('scale')
        cast_one_hot = self.additional_args.get('cast_one_hot')
        ds = VectorDataset(data_transforms, label_transforms, path="%s/%s" % (self.dir, fn), use="test",
                           cast_one_hot=cast_one_hot, scale=scale)
        self.length[2] = len(ds)
        self.raw_data_shape = Shape([ds.get_input_len()])
        return ds
