"""
simple toy data sets mostly to ensure that networks can handle the input/output
"""

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from uninas.data.abstract import AbstractDataSet
from uninas.utils.shape import Shape
from uninas.utils.args import Argument
from uninas.register import Register


class SumToyDataset(Dataset):
    def __init__(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose,
                 rows=1000, columns=10, low=0, high=10):
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms
        self.data = np.random.randint(low=low, high=high, size=(rows, columns), dtype=np.int32).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        return self.data_transforms(s), self.label_transforms(np.sum(s).reshape([1]))


@Register.data_set(regression=True)
class SumToyData(AbstractDataSet):
    """
    Toy data set of data points: input vector of length 10 and its sum as target
    """

    length = (10000, 0, 5000)  # training, valid, test
    raw_data_shape = Shape([10])
    raw_label_shape = Shape([1])
    data_mean = (0.0,)
    data_std = (1.0,)

    def _before_loading(self):
        """ called before loading training/validation/test data """
        # change the data shape of this class
        self.raw_data_shape.shape[0] = self.additional_args.get('vector_size')

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('vector_size', default=10, type=int, help='size of the vectors'),
        ]

    def _get_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return SumToyDataset(data_transforms, label_transforms,
                             rows=self.length[0], columns=self.raw_data_shape.num_features())

    def _get_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose):
        return SumToyDataset(data_transforms, label_transforms,
                             rows=self.length[2], columns=self.raw_data_shape.num_features())
