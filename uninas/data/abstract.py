import os
from typing import Union
from collections import defaultdict
from enum import Enum
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets.fakedata import FakeData
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from uninas.utils.torch.loader import InfIterator, MultiLoader, InterleavedLoader
from uninas.utils.args import ArgsInterface, MetaArgument, Argument, Namespace, find_in_args
from uninas.utils.paths import replace_standard_paths
from uninas.utils.shape import Shape
from uninas.utils.loggers.python import LoggerManager
from uninas.register import Register


class DataType(Enum):
    NONE = 0
    IMAGES2D = 1


class AbstractBatchAugmentation:
    """ applies an operation on the current loader batch """

    def __call__(self, data: torch.Tensor, label: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError


class BatchAugmentations:
    """ applies AbstractBatchAugmentations on the current loader batch """

    def __init__(self, batch_functions: [AbstractBatchAugmentation]):
        self.batch_functions = batch_functions

    def __call__(self, data: torch.Tensor, labels: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        for b in self.batch_functions:
            data, labels = b(data, labels)
        return data, labels


class AbstractDataSet(ArgsInterface):
    length = (0, 0, 0)  # training, valid, test
    data_raw_shape = Shape([])
    label_shape = Shape([])
    data_mean = None
    data_std = None

    can_download = True

    def __init__(self, data_dir: str, save_dir: Union[str, None],
                 bs_train: int, bs_test: int,
                 train_transforms: transforms.Compose, test_transforms: transforms.Compose,
                 train_batch_aug: Union[BatchAugmentations, None], test_batch_aug: Union[BatchAugmentations, None],
                 num_workers: int, num_prefetch: int,
                 valid_split: Union[int, float], valid_shuffle: bool,
                 fake: bool, download: bool,
                 **additional_args):
        """

        :param data_dir: where to find (or download) the data set
        :param save_dir: global save dir, can store and reuse the info which data was used in the random valid split
        :param bs_train: batch size for the train loader
        :param bs_test: batch size for the test loader, <= 0 to have the same as bs_train
        :param train_transforms: train augmentations (on each data point individually)
        :param test_transforms: test augmentations (on each data point individually)
        :param train_batch_aug: train augmentations (across the entire batch)
        :param test_batch_aug: test augmentations (across the entire batch)
        :param num_workers: number of workers prefetching data
        :param num_prefetch: number of batches prefetched by every worker
        :param valid_split: absolute number of data points if int or >1, otherwise a fraction of the training set
        :param valid_shuffle: whether to shuffle validation data
        :param fake: use fake data instead (no need to provide either real data or enabling downloading)
        :param download: whether downloading is allowed
        :param additional_args: arguments that are added and used by child classes
        """
        super().__init__()
        logger = LoggerManager().get_logger()
        self.dir = data_dir
        self.bs_train = bs_train
        self.bs_test = bs_test if bs_test > 0 else self.bs_train
        self.num_workers, self.num_prefetch = num_workers, num_prefetch
        self.valid_shuffle = valid_shuffle
        self.additional_args = additional_args

        self.fake = fake
        self.download = download and not self.fake
        if self.download and (not self.can_download):
            LoggerManager().get_logger().warning("The dataset can not be downloaded, but may be asked to.")

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.train_batch_augmentations = train_batch_aug
        self.test_batch_augmentations = test_batch_aug

        # load/create meta info dict
        if isinstance(save_dir, str) and len(save_dir) > 0:
            meta_path = '%s/data.meta.pt' % replace_standard_paths(save_dir)
            if os.path.isfile(meta_path):
                meta = torch.load(meta_path)
            else:
                meta = defaultdict(dict)
        else:
            meta, meta_path = defaultdict(dict), None

        # give subclasses a good spot to react to additional arguments
        self._before_loading()

        # data
        if self.fake:
            train_data = self._get_fake_train_data(self.train_transforms)
            self.test_data = self._get_fake_test_data(self.test_transforms)
        else:
            train_data = self._get_train_data(self.train_transforms)
            self.test_data = self._get_test_data(self.test_transforms)

        # split train into train+valid or using stand-alone valid set
        if valid_split > 0:
            s1 = int(valid_split) if valid_split >= 1 else int(len(train_data)*valid_split)
            if s1 >= len(train_data):
                logger.warning("Tried to set valid split larger than the training set size, setting to 0.5")
                s1 = len(train_data)//2
            s0 = len(train_data) - s1
            if meta['splits'].get((s0, s1), None) is None:
                meta['splits'][(s0, s1)] = torch.randperm(s0+s1).tolist()
            indices = meta['splits'][(s0, s1)]
            self.valid_data = torch.utils.data.Subset(train_data, np.array(indices[s0:]).astype(np.int))
            train_data = torch.utils.data.Subset(train_data, np.array(indices[0:s0]).astype(np.int))
            logger.info('Data Set: splitting training set, will use %s data points as validation set' % s1)
            if self.length[1] > 0:
                logger.info('Data Set: a dedicated validation set exists, but it will be replaced.')
        elif self.length[1] > 0:
            if self.fake:
                self.valid_data = self._get_fake_valid_data(self.test_transforms)
            else:
                self.valid_data = self._get_valid_data(self.test_transforms)
            logger.info('Data Set: using the dedicated validation set with test augmentations')
        else:
            self.valid_data = None
            logger.info('Data Set: not using a validation set at all.')
        self.train_data = train_data

        # shapes
        data, label = self.train_data[0]
        self.data_shape = Shape(list(data.shape))

        # save meta info dict
        if meta_path is not None:
            torch.save(meta, meta_path)

    def _before_loading(self):
        """ called before loading training/validation/test data """
        pass

    @classmethod
    def from_args(cls, args: Namespace, index: int = None) -> 'AbstractDataSet':
        # parsed arguments, and the global save dir
        all_args = cls._all_parsed_arguments(args, index=index)

        data_dir = replace_standard_paths(all_args.pop('dir'))
        fake = all_args.pop('fake')
        download = all_args.pop('download') and not fake

        try:
            _, save_dir = find_in_args(args, '.save_dir')
            save_dir = replace_standard_paths(save_dir)
        except ValueError:
            save_dir = ""

        # augmentations per data point and batch, for training and test
        tr_d, tr_b, te_d, te_b = [], [], [], []
        for i, aug_set in enumerate(cls._parsed_meta_arguments(Register.augmentation_sets, 'cls_augmentations', args, index=index)):
            tr_d_, tr_b_ = aug_set.get_train_transforms(args, i, cls)
            te_d_, te_b_ = aug_set.get_test_transforms(args, i, cls)
            tr_d.extend(tr_d_)
            tr_b.extend(tr_b_)
            te_d.extend(te_d_)
            te_b.extend(te_b_)
        if cls.is_on_images():
            final_transforms = [transforms.ToTensor(), transforms.Normalize(cls.data_mean, cls.data_std)]
        else:
            final_transforms = []
        train_transforms = transforms.Compose(tr_d + final_transforms)
        test_transforms = transforms.Compose(te_d + final_transforms)
        train_batch_aug = BatchAugmentations(tr_b) if len(tr_b) > 0 else None
        test_batch_aug = BatchAugmentations(te_b) if len(te_b) > 0 else None

        return cls(data_dir=data_dir, save_dir=save_dir,
                   bs_train=all_args.pop('batch_size_train'), bs_test=all_args.pop('batch_size_test'),
                   train_transforms=train_transforms, test_transforms=test_transforms,
                   train_batch_aug=train_batch_aug, test_batch_aug=test_batch_aug,
                   num_workers=all_args.pop('num_workers'), num_prefetch=all_args.pop('num_prefetch'),
                   valid_split=all_args.pop('valid_split'), valid_shuffle=all_args.pop('valid_shuffle'),
                   fake=fake, download=download, **all_args)

    def list_train_transforms(self) -> str:
        bfs = [] if self.train_batch_augmentations is None else self.train_batch_augmentations.batch_functions
        return ', '.join([cls.__class__.__name__ for cls in self.train_transforms.transforms + bfs])

    def list_test_transforms(self) -> str:
        bfs = [] if self.test_batch_augmentations is None else self.test_batch_augmentations.batch_functions
        return ', '.join([cls.__class__.__name__ for cls in self.test_transforms.transforms + bfs])

    def get_transforms(self, train=True, exclude_normalize=False) -> transforms.Compose:
        """
        get the transforms of training or test data,
        if 'exclude_normalize' is set, discard Normalize (good for visualization)
        """
        transform = self.train_transforms.transforms if train else self.test_transforms.transforms
        if exclude_normalize:
            return transforms.Compose([t for t in transform if not isinstance(t, transforms.Normalize)])
        return transform

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        kwargs = Register.get_my_kwargs(cls)
        aug_sets = Register.augmentation_sets.filter_match_all(on_images=kwargs.get('images'))
        return super().meta_args_to_add() + [
            MetaArgument('cls_augmentations', aug_sets, help_name='data augmentation'),
        ]

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        args = super().args_to_add(index) + [
            Argument('dir', default='{path_data}', type=str, help='data dir', is_path=True),
            Argument('download', default='False', type=str, help='allow downloading', is_bool=True),
            Argument('fake', default='False', type=str, help='use fake data', is_bool=True),
            Argument('batch_size_train', default=64, type=int, help='batch size for each train data loader (i.e. ddp may cause multiple instances)'),
            Argument('batch_size_test', default=-1, type=int, help='batch size for each eval/test data loader, same as train size if <0'),
            Argument('num_workers', default=4, type=int, help='number of workers for data loaders'),
            Argument('num_prefetch', default=2, type=int, help='number batches that each worker prefetches'),
            Argument('valid_split', default=0.0, type=float, help='num samples if >1, else % split, for the validation set'),
            Argument('valid_shuffle', default='False', type=str, help='shuffle the validation set', is_bool=True),
        ]
        return args

    def get_batch_size(self, train=True) -> int:
        return self.bs_train if train else self.bs_test

    def get_data_shape(self) -> Shape:
        return self.data_shape

    def get_label_shape(self) -> Shape:
        return self.__class__.label_shape

    def _str_dict(self) -> dict:
        dct = super()._str_dict()
        dct.update({
            'data shape': self.get_data_shape(),
            'label shape': self.get_label_shape(),
        })
        dct.update({'fake': self.fake} if self.fake else {})
        dct.update({} if self.train_data is None else {'training data': len(self.train_data)})
        dct.update({} if self.valid_data is None else {'valid data': len(self.valid_data)})
        dct.update({} if self.test_data is None else {'test data': len(self.test_data)})
        dct.update({} if self.test_data is None else {'train batch size': self.bs_train})
        dct.update({} if self.test_data is None else {'test batch size': self.bs_test})
        return dct

    def sample_random_data(self, batch_size=1) -> torch.Tensor:
        """ get random data with correct size """
        size = [batch_size] + list(self.data_shape.shape)
        return torch.randn(size=size, dtype=torch.float32)

    def train_loader(self, dist=False) -> InfIterator:
        return self._loader(self.train_data, is_train=True, shuffle=True, dist=dist, wrap=True)

    def valid_loader(self, dist=False) -> InfIterator:
        return self._loader(self.valid_data, is_train=False, shuffle=self.valid_shuffle, dist=dist, wrap=True)

    def mixed_train_valid_loader(self, dist=False) -> MultiLoader:
        """ for having training/valid set both for training, in bi-optimization settings """
        assert self.train_data is not None, 'Training data must not be None when using mixed loading'
        assert self.valid_data is not None, 'Valid data must not be None when using mixed loading'
        return MultiLoader([
            self._loader(self.train_data, is_train=True, shuffle=True, dist=dist, wrap=False),
            self._loader(self.valid_data, is_train=True, shuffle=self.valid_shuffle, dist=dist, wrap=False)
        ])

    def interleaved_train_valid_loader(self, multiples=(1, 1), dist=False) -> InterleavedLoader:
        """ for having training/valid set both for training, in bi-optimization settings """
        assert self.train_data is not None, 'Training data must not be None when using mixed loading'
        assert self.valid_data is not None, 'Valid data must not be None when using mixed loading'
        return InterleavedLoader([
            self._loader(self.train_data, is_train=True, shuffle=True, dist=dist, wrap=False),
            self._loader(self.valid_data, is_train=True, shuffle=self.valid_shuffle, dist=dist, wrap=False)
        ], multiples)

    def test_loader(self, dist=False) -> InfIterator:
        return self._loader(self.test_data, is_train=False, shuffle=False, dist=dist, wrap=True)

    def _loader(self, data, is_train=True, shuffle=True, dist=False, wrap=True):
        if data is None:
            return None
        bs = self.bs_train if is_train else self.bs_test
        sampler = DistributedSampler(data) if dist else None
        loader = DataLoader(data, batch_size=bs, shuffle=shuffle and not dist, num_workers=self.num_workers,
                            pin_memory=True, sampler=sampler, prefetch_factor=self.num_prefetch)
        if wrap:
            return InfIterator(loader)
        return loader

    def _get_train_data(self, used_transforms: transforms.Compose):
        raise NotImplementedError

    def _get_valid_data(self, used_transforms: transforms.Compose):
        return None

    def _get_test_data(self, used_transforms: transforms.Compose):
        raise NotImplementedError

    def _get_fake_train_data(self, used_transforms: transforms.Compose):
        raise NotImplementedError

    def _get_fake_valid_data(self, used_transforms: transforms.Compose):
        raise NotImplementedError

    def _get_fake_test_data(self, used_transforms: transforms.Compose):
        raise NotImplementedError

    @classmethod
    def is_on_images(cls) -> bool:
        kwargs = Register.get_my_kwargs(cls)
        return kwargs.get('images', False)

    @classmethod
    def is_classification(cls) -> bool:
        kwargs = Register.get_my_kwargs(cls)
        return kwargs.get('classification', False)

    @classmethod
    def num_classes(cls) -> int:
        kwargs = Register.get_my_kwargs(cls)
        assert kwargs.get('classification', False)
        return cls.label_shape.num_features()


class AbstractCNNClassificationDataSet(AbstractDataSet):
    length = (0, 0, 0)  # training, valid, test
    data_raw_shape = Shape([-1, -1, -1])  # channel height width
    label_shape = Shape([-1])
    data_mean = (-1, -1, -1)
    data_std = (-1, -1, -1)

    def _get_train_data(self, used_transforms: transforms.Compose) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def _get_test_data(self, used_transforms: transforms.Compose) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def _get_fake_train_data(self, used_transforms: transforms.Compose) -> torch.utils.data.Dataset:
        return FakeData(self.length[0], self.data_raw_shape.shape, self.num_classes(), used_transforms)

    def _get_fake_valid_data(self, used_transforms: transforms.Compose) -> torch.utils.data.Dataset:
        return FakeData(self.length[1], self.data_raw_shape.shape, self.num_classes(), used_transforms)

    def _get_fake_test_data(self, used_transforms: transforms.Compose) -> torch.utils.data.Dataset:
        return FakeData(self.length[2], self.data_raw_shape.shape, self.num_classes(), used_transforms)


class AbstractAug(ArgsInterface):
    """ a set of transforms (per data point) and batch augmentations """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('force_type', default='', type=str, help='force use train/test augmentation if set',
                     choices=['', 'train', 'test']),
        ]

    @classmethod
    def get_train_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (list, [BatchAugmentations]):
        """
        :param args: global argparse namespace
        :param index: index of this AbstractAug
        :param ds: dataset class (has e.g. shape info)
        :returns
            a list of transforms for individual data points
            a list of BatchAugmentations for batches
        """
        if cls._parsed_argument('force_type', args, index=index) == 'test':
            return cls._get_test_transforms(args, index, ds)
        return cls._get_train_transforms(args, index, ds)

    @classmethod
    def _get_train_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (list, [BatchAugmentations]):
        """
        :param args: global argparse namespace
        :param index: index of this AbstractAug
        :param ds: dataset class (has e.g. shape info)
        :returns
            a list of transforms for individual data points
            a list of BatchAugmentations for batches
        """
        raise NotImplementedError

    @classmethod
    def get_test_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (list, [BatchAugmentations]):
        """
        :param args: global argparse namespace
        :param index: index of this AbstractAug
        :param ds: dataset class (has e.g. shape info)
        :returns
            a list of transforms for individual data points
            a list of BatchAugmentations for batches
        """
        if cls._parsed_argument('force_type', args, index=index) == 'train':
            return cls._get_train_transforms(args, index, ds)
        return cls._get_test_transforms(args, index, ds)

    @classmethod
    def _get_test_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (list, [BatchAugmentations]):
        """
        :param args: global argparse namespace
        :param index: index of this AbstractAug
        :param ds: dataset class (has e.g. shape info)
        :returns
            a list of transforms for individual data points
            a list of BatchAugmentations for batches
        """
        raise NotImplementedError
