import os
from typing import Union, List, Any
from collections import defaultdict
from enum import Enum
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets.fakedata import FakeData
from torch.utils.data import Dataset
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


class AugType(Enum):
    NONE = 0            # not to be applied
    ON_RAW = 1          # applied before normalization
    ON_NORM = 2         # applied after normalization
    ON_BATCHES = 99     # applied not on single data points, but batches thereof

    @classmethod
    def sorted(cls, augments: {'AugType', list}, final_default: list) -> list:
        assert len(augments[AugType.NONE]) == 0, "Some augmentations are applied to None, but not empty"
        in_order = []
        in_order.extend(augments[AugType.ON_RAW])
        in_order.extend(final_default)
        in_order.extend(augments[AugType.ON_NORM])
        # in_order.extend(augments[AugType.ON_BATCHES])  # this is purposefully left out, as transforms are on singles
        return in_order


class AbstractBatchAugmentation:
    """ applies an operation on the current loader batch """

    def __call__(self, data: torch.Tensor, label: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError


class CastToTensor:
    def __call__(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x)


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
    raw_data_shape = Shape([])
    raw_label_shape = Shape([])
    data_mean = None
    data_std = None

    can_download = True

    def __init__(self, data_dir: str, save_dir: Union[str, None],
                 batch_size_train: int, batch_size_test: int,
                 train_transforms: transforms.Compose, test_transforms: transforms.Compose,
                 train_label_transforms: transforms.Compose, test_label_transforms: transforms.Compose,
                 train_batch_aug: Union[BatchAugmentations, None], test_batch_aug: Union[BatchAugmentations, None],
                 num_workers=8, num_prefetch=2,
                 valid_split: Union[int, float] = 0, valid_shuffle=True, valid_as_test=False,
                 fake=False, download=False,
                 **additional_args):
        """

        :param data_dir: where to find (or download) the data set
        :param save_dir: global save dir, can store and reuse the info which data was used in the random valid split
        :param batch_size_train: batch size for the train loader
        :param batch_size_test: batch size for the test loader, <= 0 to have the same as bs_train
        :param train_transforms: train augmentations (on each data point individually)
        :param test_transforms: test augmentations (on each data point individually)
        :param train_label_transforms: train augmentations for labels (on each data point individually)
        :param test_label_transforms: test augmentations for labels (on each data point individually)
        :param train_batch_aug: train augmentations (across the entire batch)
        :param test_batch_aug: test augmentations (across the entire batch)
        :param num_workers: number of workers prefetching data
        :param num_prefetch: number of batches prefetched by every worker
        :param valid_split: absolute number of data points if int or >1, otherwise a fraction of the training set
        :param valid_shuffle: whether to shuffle validation data
        :param valid_as_test: use validation data as test data instead (only true valid data, not train -> valid)
        :param fake: use fake data instead (no need to provide either real data or enabling downloading)
        :param download: whether downloading is allowed
        :param additional_args: arguments that are added and used by child classes
        """
        super().__init__()
        logger = LoggerManager().get_logger()
        self.dir = data_dir
        self.bs_train = batch_size_train
        self.bs_test = batch_size_test if batch_size_test > 0 else self.bs_train
        self.num_workers, self.num_prefetch = num_workers, num_prefetch
        self.valid_shuffle = valid_shuffle
        self.additional_args = additional_args

        self.fake = fake
        self.download = download and not self.fake
        if self.download and (not self.can_download):
            LoggerManager().get_logger().warning("The dataset can not be downloaded, but may be asked to.")

        self.train_transforms = train_transforms
        self.valid_transforms = test_transforms
        self.test_transforms = test_transforms
        self.train_label_transforms = train_label_transforms
        self.valid_label_transforms = test_label_transforms
        self.test_label_transforms = test_label_transforms
        self.train_batch_augmentations = train_batch_aug
        self.valid_batch_augmentations = test_batch_aug
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

        # train, valid and test data
        if self.fake:
            train_data = self._get_fake_train_data(self.train_transforms, self.train_label_transforms)
            self.valid_data = self._get_fake_valid_data(self.valid_transforms, self.valid_label_transforms)
            self.test_data = self._get_fake_test_data(self.test_transforms, self.test_label_transforms)
        else:
            train_data = self._get_train_data(self.train_transforms, self.train_label_transforms)
            self.valid_data = self._get_valid_data(self.valid_transforms, self.valid_label_transforms)
            self.test_data = self._get_test_data(self.test_transforms, self.test_label_transforms)

        # possibly use valid as test data instead
        if valid_as_test:
            assert self.test_data is None, "Can not use valid as test data if test data is available"
            assert self.valid_data is not None, "Can not use valid as test data if no valid data is available"
            self.test_data = self.valid_data
            self.valid_data = None
            logger.info('Data Set: using the dedicated validation set as test set instead')

        # split train into train+valid or using stand-alone valid set
        if valid_split > 0:
            assert self.valid_data is None, "Can not use training as valid data if valid data is available"
            assert train_data is not None, "Can not use training as valid data if no training data is available"
            self.valid_transforms = self.train_transforms
            self.valid_label_transforms = self.train_label_transforms
            self.valid_batch_augmentations = self.train_batch_augmentations

            s1 = int(valid_split) if valid_split >= 1 else int(self._get_len(train_data)*valid_split)
            if s1 >= self._get_len(train_data):
                logger.warning("Tried to set valid split larger than the training set size, setting to 0.5")
                s1 = self._get_len(train_data)//2
            s0 = self._get_len(train_data) - s1
            if meta['splits'].get((s0, s1), None) is None:
                meta['splits'][(s0, s1)] = torch.randperm(s0+s1).tolist()
            indices = meta['splits'][(s0, s1)]
            self.valid_data = torch.utils.data.Subset(train_data, np.array(indices[s0:]).astype(np.int32))
            train_data = torch.utils.data.Subset(train_data, np.array(indices[0:s0]).astype(np.int32))
            logger.info('Data Set: splitting the training set, will use %s data points as validation set' % s1)
        self.train_data = train_data

        # shapes
        try:
            data, label = self.train_data[0]
            self.data_shape = Shape(list(data.shape))
            self.label_shape = Shape(list(label.shape))
            if self.is_classification():
                if self.label_shape.num_dims() == 0:
                    self.label_shape = Shape([1])
                self.label_shape.shape[0] = self.num_classes()
        except KeyError as e:
            LoggerManager().get_logger().error("Data Set: Error. DataSet and Augmentations seem to not fit each other")
            raise e

        # save meta info dict
        if meta_path is not None:
            torch.save(meta, meta_path)

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
        aug_train, aug_test = defaultdict(list), defaultdict(list)
        for i, aug_set in enumerate(cls._parsed_meta_arguments(Register.augmentation_sets, 'cls_augmentations', args, index=index)):
            aug_type, augments = aug_set.get_train_transforms(args, i, cls)
            aug_train[aug_type].extend(augments)
            aug_type, augments = aug_set.get_test_transforms(args, i, cls)
            aug_test[aug_type].extend(augments)

        # add normalization for images, compose
        train_transforms = cls._compose_data_transforms(aug_train)
        test_transforms = cls._compose_data_transforms(aug_test)

        # maybe use batch augmentations
        train_batch_aug, test_batch_aug = None, None
        if len(aug_train[AugType.ON_BATCHES]) > 0:
            train_batch_aug = BatchAugmentations(aug_train[AugType.ON_BATCHES])
        if len(aug_train[AugType.ON_BATCHES]) > 0:
            test_batch_aug = BatchAugmentations(aug_test[AugType.ON_BATCHES])

        # maybe use label augmentations
        train_label_transforms = cls._compose_label_transforms(defaultdict(list))
        test_label_transforms = cls._compose_label_transforms(defaultdict(list))

        return cls(data_dir=data_dir, save_dir=save_dir,
                   train_transforms=train_transforms, test_transforms=test_transforms,
                   train_label_transforms=train_label_transforms, test_label_transforms=test_label_transforms,
                   train_batch_aug=train_batch_aug, test_batch_aug=test_batch_aug,
                   fake=fake, download=download, **all_args)

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
            Argument('valid_as_test', default='False', type=str, help='use validation as test data instead', is_bool=True),
        ]
        return args

    def _before_loading(self):
        """ called before loading training/validation/test data """
        pass

    @classmethod
    def _compose_data_transforms(cls, augments: {AugType, list}) -> transforms.Compose:
        final_default = []
        if cls.is_on_images() and (None not in [cls.data_mean, cls.data_std]):
            final_default.extend([transforms.ToTensor(), transforms.Normalize(cls.data_mean, cls.data_std)])
        in_order = AugType.sorted(augments, final_default)
        return transforms.Compose(in_order)

    @classmethod
    def _compose_label_transforms(cls, augments: {AugType, list}) -> transforms.Compose:
        final_default = [CastToTensor()]
        in_order = AugType.sorted(augments, final_default)
        return transforms.Compose(in_order)

    @staticmethod
    def _list_transforms(augmentations: Union[transforms.Compose, None],
                         batch_augmentations: Union[BatchAugmentations, None]) -> str:
        if (augmentations is None) and (batch_augmentations is None):
            return "no transforms"
        bfs = [] if batch_augmentations is None else batch_augmentations.batch_functions
        try:
            return ', '.join([cls.__class__.__name__ for cls in augmentations.transforms + bfs])
        except:
            return 'unknown transforms'

    def list_train_transforms(self) -> str:
        return self._list_transforms(self.train_transforms, self.train_batch_augmentations)

    def list_valid_transforms(self) -> str:
        return self._list_transforms(self.valid_transforms, self.valid_batch_augmentations)

    def list_test_transforms(self) -> str:
        return self._list_transforms(self.test_transforms, self.test_batch_augmentations)

    def get_data_transforms(self, train=True, exclude_normalize=False) -> transforms.Compose:
        """
        get the transforms of training or test data,
        if 'exclude_normalize' is set, discard Normalize (good for visualization)
        """
        transform = self.train_transforms.transforms if train else self.test_transforms.transforms
        if exclude_normalize:
            return transforms.Compose([t for t in transform if not isinstance(t, transforms.Normalize)])
        return transform

    def get_batch_size(self, train=True) -> int:
        return self.bs_train if train else self.bs_test

    def get_data_shape(self) -> Shape:
        return self.data_shape

    def get_label_shape(self) -> Shape:
        return self.label_shape

    @staticmethod
    def _get_len(data: Any) -> int:
        if data is None:
            return 0
        try:
            return len(data)
        except:
            raise AttributeError("data of type %s does not have a length" % type(data))

    def _str_dict(self) -> dict:
        dct = super()._str_dict()
        dct.update({
            'data shape': self.get_data_shape(),
            'label shape': self.get_label_shape(),
        })
        dct.update({'fake': self.fake} if self.fake else {})
        dct.update({} if self.train_data is None else {'training data': self._get_len(self.train_data)})
        dct.update({} if self.valid_data is None else {'valid data': self._get_len(self.valid_data)})
        dct.update({} if self.test_data is None else {'test data': self._get_len(self.test_data)})
        dct.update({} if self.test_data is None else {'train batch size': self.bs_train})
        dct.update({} if self.test_data is None else {'test batch size': self.bs_test})
        return dct

    def sample_random_data(self, batch_size=1) -> torch.Tensor:
        """ get random data with correct size """
        size = [batch_size] + list(self.data_shape.shape)
        return torch.randn(size=size, dtype=torch.float32)

    def sample_random_label(self, batch_size=1) -> torch.Tensor:
        """ get random labels with correct size """
        size = [batch_size] + list(self.label_shape.shape)
        dtype = torch.long if self.is_classification() else torch.float32
        return torch.randn(size=size, dtype=dtype)

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

    def _get_full_data(self, data: Union[None, Dataset], to_numpy=False, num=-1)\
            -> (Union[None, torch.Tensor, np.array], Union[None, torch.Tensor, np.array]):
        """
        concatenates up to num data samples into one bulk

        :param data: dataset to use
        :param to_numpy: cast inputs and targets to numpy arrays, otherwise cast them to tensors
        :param num: max number of samples, use the data set length if <=0
        """
        if data is None:
            return None, None
        assert isinstance(data, Dataset)
        num = self._get_len(data) if num <= 0 else min([self._get_len(data), num])

        # get the data
        bulk_data, bulk_labels = [], []
        for i in range(num):
            x, y = data[i]
            bulk_data.append(x)
            bulk_labels.append(y)

        # stack
        if isinstance(bulk_data[0], torch.Tensor):
            bulk_data = torch.stack(bulk_data, dim=0)
        else:
            bulk_data = np.stack(bulk_data, axis=0)
        if isinstance(bulk_labels[0], torch.Tensor):
            bulk_labels = torch.stack(bulk_labels, dim=0)
        else:
            bulk_labels = np.stack(bulk_labels, axis=0)

        # cast
        if to_numpy and isinstance(bulk_data, torch.Tensor):
            bulk_data = bulk_data.cpu().numpy()
        elif (not to_numpy) and not isinstance(bulk_data, torch.Tensor):
            bulk_data = torch.Tensor(bulk_data)
        if to_numpy and isinstance(bulk_labels, torch.Tensor):
            bulk_labels = bulk_labels.cpu().numpy()
        elif (not to_numpy) and not isinstance(bulk_labels, torch.Tensor):
            bulk_labels = torch.Tensor(bulk_labels)

        return bulk_data, bulk_labels

    def get_full_train_data(self, to_numpy=False, num=-1) -> (Union[None, torch.Tensor, np.array], Union[None, torch.Tensor, np.array]):
        """ get up to num data samples and labels in a bulk """
        return self._get_full_data(self.train_data, to_numpy=to_numpy, num=num)

    def get_full_valid_data(self, to_numpy=False, num=-1) -> (Union[None, torch.Tensor, np.array], Union[None, torch.Tensor, np.array]):
        """ get up to num data samples and labels in a bulk """
        return self._get_full_data(self.valid_data, to_numpy=to_numpy, num=num)

    def get_full_test_data(self, to_numpy=False, num=-1) -> (Union[None, torch.Tensor, np.array], Union[None, torch.Tensor, np.array]):
        """ get up to num data samples and labels in a bulk """
        return self._get_full_data(self.test_data, to_numpy=to_numpy, num=num)

    def _get_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose)\
            -> Union[torch.utils.data.Dataset, None]:
        return None

    def _get_valid_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose)\
            -> Union[torch.utils.data.Dataset, None]:
        return None

    def _get_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose)\
            -> Union[torch.utils.data.Dataset, None]:
        return None

    def _get_fake_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose)\
            -> Union[torch.utils.data.Dataset, None]:
        return None

    def _get_fake_valid_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose)\
            -> Union[torch.utils.data.Dataset, None]:
        return None

    def _get_fake_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose)\
            -> Union[torch.utils.data.Dataset, None]:
        return None

    def undo_label_normalization(self, labels: Union[torch.Tensor, np.array]) -> Union[torch.Tensor, np.array]:
        """
        Undo possible normalization for the labels
        :param labels: [batch, ...]
        """
        assert self.is_regression()
        return labels

    @classmethod
    def is_on_images(cls) -> bool:
        return cls.matches_registered_properties(images=True)

    @classmethod
    def is_classification(cls) -> bool:
        return cls.matches_registered_properties(classification=True)

    @classmethod
    def is_regression(cls) -> bool:
        return cls.matches_registered_properties(regression=True)

    @classmethod
    def num_classes(cls) -> int:
        assert cls.is_classification()
        return cls.raw_label_shape.num_features()

    @classmethod
    def get_class_names(cls) -> Union[List[str], None]:
        assert cls.is_classification()
        return None


class AbstractCNNClassificationDataSet(AbstractDataSet):
    length = (0, 0, 0)  # training, valid, test
    raw_data_shape = Shape([-1, -1, -1])  # channel height width
    raw_label_shape = Shape([-1])
    data_mean = (-1, -1, -1)
    data_std = (-1, -1, -1)

    def _get_fake_train_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose)\
            -> Union[torch.utils.data.Dataset, None]:
        if self.length[0] <= 0:
            return None
        return FakeData(self.length[0], image_size=self.raw_data_shape.shape, num_classes=self.num_classes(),
                        transform=data_transforms, target_transform=label_transforms)

    def _get_fake_valid_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose)\
            -> Union[torch.utils.data.Dataset, None]:
        if self.length[1] <= 0:
            return None
        return FakeData(self.length[1], image_size=self.raw_data_shape.shape, num_classes=self.num_classes(),
                        transform=data_transforms, target_transform=label_transforms)

    def _get_fake_test_data(self, data_transforms: transforms.Compose, label_transforms: transforms.Compose)\
            -> Union[torch.utils.data.Dataset, None]:
        if self.length[2] <= 0:
            return None
        return FakeData(self.length[2], image_size=self.raw_data_shape.shape, num_classes=self.num_classes(),
                        transform=data_transforms, target_transform=label_transforms)


class AbstractAug(ArgsInterface):
    """ a set of transforms (per data point) and batch augmentations """

    @classmethod
    def applies_to_labels(cls) -> bool:
        return False

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('force_type', default='', type=str, help='force use train/test augmentation if set',
                     choices=['', 'train', 'test']),
        ]

    @classmethod
    def get_train_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
        if cls._parsed_argument('force_type', args, index=index) == 'test':
            return cls._get_test_transforms(args, index, ds)
        return cls._get_train_transforms(args, index, ds)

    @classmethod
    def get_test_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
        if cls._parsed_argument('force_type', args, index=index) == 'train':
            return cls._get_train_transforms(args, index, ds)
        return cls._get_test_transforms(args, index, ds)

    @classmethod
    def _get_train_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
        """
        :param args: global argparse namespace
        :param index: index of this AbstractAug
        :param ds: dataset class (has e.g. shape info)
        :returns
            a type that clarifies when to apply the transforms
            a list of (batch) transforms
        """
        raise NotImplementedError

    @classmethod
    def _get_test_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
        """
        :param args: global argparse namespace
        :param index: index of this AbstractAug
        :param ds: dataset class (has e.g. shape info)
        :returns
            a type that clarifies when to apply the transforms
            a list of (batch) transforms
        """
        raise NotImplementedError
