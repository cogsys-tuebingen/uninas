"""
profiled data
"""

import os
import random
from typing import Union
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from uninas.data.abstract import AbstractDataSet
from uninas.utils.shape import Shape
from uninas.utils.args import Argument
from uninas.utils.np import concatenated_one_hot
from uninas.utils.paths import replace_standard_paths
from uninas.register import Register


class VectorDataset(Dataset):
    def __init__(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose,
                 keys: list, values: list):
        assert len(keys) == len(values)
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms
        self.keys = keys
        self.values = values

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.data_transforms(self.keys[idx]), self.label_transforms(self.values[idx])


class VectorDataStorage:
    def __init__(self, path: str, cast_one_hot=False, normalize_labels=False):
        self._cast_one_hot = cast_one_hot
        self._normalize_labels = normalize_labels
        all_data = torch.load(replace_standard_paths(path))
        self._sizes = all_data['sizes']
        self._data_train = all_data['train']
        self._data_test = all_data['test']
        values = list(self._data_train.values())
        self._label_mean = np.mean(values) if self._normalize_labels else 0.0
        self._label_std = np.std(values) if self._normalize_labels else 1.0

    def get_input_len(self) -> int:
        if self._cast_one_hot:
            return sum(self._sizes)
        return len(self._sizes)

    def get_label_mean_std(self) -> (float, float):
        return self._label_mean, self._label_std

    def undo_label_normalization(self, labels: Union[torch.Tensor, np.array]) -> Union[torch.Tensor, np.array]:
        """
        Undo possible normalization for the labels
        :param labels: [batch, ...]
        """
        if self._normalize_labels:
            return (labels * self._label_std) + self._label_mean
        return labels

    def _get_set(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose,
                 data: {tuple: float}, num=-1) -> VectorDataset:
        keys = list(data.keys())
        if num > 0:
            keys = keys[:num]
        values = [np.array([data[k]], dtype=np.float32) for k in keys]
        if self._normalize_labels:
            values = [(v - self._label_mean) / self._label_std for v in values]
        keys = [concatenated_one_hot(k, self._sizes, dtype=np.float32) for k in keys] if self._cast_one_hot else\
            [np.array(k, dtype=np.float32) for k in keys]
        return VectorDataset(data_transforms, label_transforms, keys, values)

    def get_train_set(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose, num=-1) -> VectorDataset:
        return self._get_set(data_transforms, label_transforms, self._data_train, num=num)

    def get_test_set(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose, num=-1) -> VectorDataset:
        return self._get_set(data_transforms, label_transforms, self._data_test, num=num)

    def get_dict(self) -> {tuple: float}:
        """ get the full data as dict """
        data = self._data_train.copy()
        data.update(self._data_test.copy())
        return data


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
    data_mean = (0.0,)              # depends on the loaded data, but not changed
    data_std = (1.0,)               # depends on the loaded data, but not changed

    def _before_loading(self):
        """ called before loading training/validation/test data """
        # change the data shape of this class
        self.all_data = VectorDataStorage(path="%s/%s" % (self.dir, self.additional_args.get('file_name')),
                                          cast_one_hot=self.additional_args.get('cast_one_hot'),
                                          normalize_labels=self.additional_args.get('normalize_labels'))
        self.raw_data_shape = Shape([self.all_data.get_input_len()])

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('file_name', default="profiled.pt", type=str, help='name of the save file'),
            Argument('train_num', default=-1, type=int, help='reduce the training set to this number of data points'),
            Argument('cast_one_hot', default="True", type=str, help='cast the training inputs to one-hot', is_bool=True),
            Argument('normalize_labels', default="False", type=str, help='normalize the label values', is_bool=True),
        ]

    def _str_dict(self) -> dict:
        dct = super()._str_dict()
        dct.update({
            'normalize labels': self.additional_args.get('normalize_labels'),
        })
        return dct

    def _get_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        num = self.additional_args.get('train_num')
        ds = self.all_data.get_train_set(data_transforms, label_transforms, num=num)
        self.length[0] = len(ds)
        return ds

    def _get_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        ds = self.all_data.get_test_set(data_transforms, label_transforms, num=-1)
        self.length[2] = len(ds)
        return ds

    def undo_label_normalization(self, labels: Union[torch.Tensor, np.array]) -> Union[torch.Tensor, np.array]:
        """
        Undo possible normalization for the labels
        :param labels: [batch, ...]
        """
        assert self.is_regression()
        return self.all_data.undo_label_normalization(labels)

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
