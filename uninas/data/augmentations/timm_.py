from torchvision import transforms
from uninas.data.abstract import AbstractDataSet, AbstractAug, AugType
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


try:
    from timm.data.transforms import RandomResizedCropAndInterpolation


    @Register.augmentation_set(on_single=True, on_images=True)
    class TimmImagenetAug(AbstractAug):
        """
        Common standard ImageNet transforms, cropping to specific size, horizontal flipping and color jitter
        """

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return super().args_to_add(index) + [
                Argument('crop_size', default=224, type=int, help='to which size to crop the images'),
                Argument('color_jitter', default=0.4, type=float, help='jitter for brightness, contrast and saturation'),
                Argument('interpolation', default='random', type=str, help='interpolation',
                         choices=['random', 'bilinear', 'bicubic', 'lanczos', 'hamming']),
                Argument('scale_min', default=0.08, type=float, help='scale min'),
                Argument('scale_max', default=1.0, type=float, help='scale max'),
                Argument('ratio_min', default=0.75, type=float, help='ratio min'),
                Argument('ratio_max', default=1.3333333333333333, type=float, help='ratio max'),
            ]

        @classmethod
        def _get_train_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
            assert ds.raw_data_shape.num_dims() == 3
            all_parsed = cls._all_parsed_arguments(args, index=index)
            all_transforms = [
                RandomResizedCropAndInterpolation(all_parsed.get('crop_size'),
                                                  scale=(all_parsed.get('scale_min'), all_parsed.get('scale_max')),
                                                  ratio=(all_parsed.get('ratio_min'), all_parsed.get('ratio_max')),
                                                  interpolation=all_parsed.get('interpolation')),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=all_parsed.get('color_jitter'),
                    contrast=all_parsed.get('color_jitter'),
                    saturation=all_parsed.get('color_jitter')),
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

except ImportError as e:
    Register.missing_import(e)
