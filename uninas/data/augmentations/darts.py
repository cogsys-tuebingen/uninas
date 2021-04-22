from torchvision import transforms
from uninas.data.abstract import AbstractDataSet, AbstractAug, AugType
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


@Register.augmentation_set(on_single=True, on_images=True)
class DartsCifarAug(AbstractAug):
    """
    Common CIFAR transforms, random cropping (pad=4) and horizontal flipping
    """

    @classmethod
    def _get_train_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
        assert ds.raw_data_shape.num_dims() == 3
        all_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        return AugType.ON_RAW, all_transforms

    @classmethod
    def _get_test_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
        assert ds.raw_data_shape.num_dims() == 3
        return AugType.NONE, []


@Register.augmentation_set(on_single=True, on_images=True)
class DartsImagenetAug(AbstractAug):
    """
    Common standard ImageNet transforms, cropping to specific size, horizontal flipping and color jitter
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('crop_size', default=224, type=int, help='to which size to crop the images'),
        ]

    @classmethod
    def _get_train_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
        assert ds.raw_data_shape.num_dims() == 3
        crop_size = cls._parsed_argument('crop_size', args, index=index)
        all_transforms = [
            transforms.RandomResizedCrop(crop_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
        ]
        return AugType.ON_RAW, all_transforms

    @classmethod
    def _get_test_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
        assert ds.raw_data_shape.num_dims() == 3
        crop_size = cls._parsed_argument('crop_size', args, index=index)
        all_transforms = [
            transforms.Resize(int(crop_size / 0.875)),
            transforms.CenterCrop(crop_size),
        ]
        return AugType.ON_RAW, all_transforms
