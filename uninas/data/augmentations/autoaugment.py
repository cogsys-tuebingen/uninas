"""
AutoAugment: Learning Augmentation Policies from Data
https://arxiv.org/abs/1805.09501v1

implementation adapted from https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
"""

import random
import numpy as np
from torchvision import transforms
from uninas.data.abstract import AbstractDataSet, AbstractAug, AugType
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


try:
    from PIL import Image, ImageEnhance, ImageOps


    class SinglePolicy:
        magnitudes = None

        def __init__(self, m=0, p=0.5, fill_color=(0, 0, 0)):
            self.magnitude = self.__class__.magnitudes[m]
            self.fill_color = fill_color
            self.probability = p

        def __call__(self, img: Image) -> Image:
            if random.random() < self.probability:
                return self._apply(img)
            return img

        def _apply(self, img: Image) -> Image:
            raise NotImplementedError

        @staticmethod
        def random_sign() -> int:
            return random.choice([1, -1])


    class ShearX(SinglePolicy):
        magnitudes = np.linspace(0, 0.3, 10)

        def _apply(self, img: Image) -> Image:
            return img.transform(img.size, Image.AFFINE,
                                 (1, self.magnitude * self.random_sign(), 0, 0, 1, 0),
                                 Image.BICUBIC, fillcolor=self.fill_color)


    class ShearY(SinglePolicy):
        magnitudes = np.linspace(0, 0.3, 10)

        def _apply(self, img: Image) -> Image:
            return img.transform(img.size, Image.AFFINE,
                                 (1, 0, 0, self.magnitude * self.random_sign(), 1, 0),
                                 Image.BICUBIC, fillcolor=self.fill_color)


    class TranslateX(SinglePolicy):
        magnitudes = np.linspace(0, 150 / 331, 10)

        def _apply(self, img: Image) -> Image:
            return img.transform(img.size, Image.AFFINE,
                                 (1, 0, self.magnitude * self.random_sign() * img.size[0], 0, 1, 0),
                                 fillcolor=self.fill_color)


    class TranslateY(SinglePolicy):
        magnitudes = np.linspace(0, 150 / 331, 10)

        def _apply(self, img: Image) -> Image:
            return img.transform(img.size, Image.AFFINE,
                                 (1, 0, 0, 0, 1, self.magnitude * self.random_sign() * img.size[1]),
                                 fillcolor=self.fill_color)


    class Rotate(SinglePolicy):
        magnitudes = np.linspace(0, 30, 10)

        def _apply(self, img: Image) -> Image:
            rot = img.convert("RGBA").rotate(self.magnitude * self.random_sign())
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)


    class Color(SinglePolicy):
        magnitudes = np.linspace(0.0, 0.9, 10)

        def _apply(self, img: Image) -> Image:
            return ImageEnhance.Color(img).enhance(1 + self.magnitude * self.random_sign())


    class Posterize(SinglePolicy):
        magnitudes = np.round(np.linspace(8, 4, 10), 0).astype(np.int32)

        def _apply(self, img: Image) -> Image:
            return ImageOps.posterize(img, self.magnitude)


    class Solarize(SinglePolicy):
        magnitudes = np.linspace(256, 0, 10)

        def _apply(self, img: Image) -> Image:
            return ImageOps.solarize(img, self.magnitude)


    class Contrast(SinglePolicy):
        magnitudes = np.linspace(0.0, 0.9, 10)

        def _apply(self, img: Image) -> Image:
            return ImageEnhance.Contrast(img).enhance(1 + self.magnitude * self.random_sign())


    class Sharpness(SinglePolicy):
        magnitudes = np.linspace(0.0, 0.9, 10)

        def _apply(self, img: Image) -> Image:
            return ImageEnhance.Sharpness(img).enhance(1 + self.magnitude * self.random_sign())


    class Brightness(SinglePolicy):
        magnitudes = np.linspace(0.0, 0.9, 10)

        def _apply(self, img: Image) -> Image:
            return ImageEnhance.Brightness(img).enhance(1 + self.magnitude * self.random_sign())


    class AutoContrast(SinglePolicy):
        magnitudes = [0]*10

        def _apply(self, img: Image) -> Image:
            return ImageOps.autocontrast(img)


    class Equalize(SinglePolicy):
        magnitudes = [0]*10

        def _apply(self, img: Image) -> Image:
            return ImageOps.equalize(img)


    class Invert(SinglePolicy):
        magnitudes = [0]*10

        def _apply(self, img: Image) -> Image:
            return ImageOps.invert(img)


    class SubPolicy:
        def __init__(self, p0: SinglePolicy, p1: SinglePolicy):
            self.p0 = p0
            self.p1 = p1

        def __call__(self, img: Image):
            return self.p1(self.p0(img))


    class RandomSubPolicies:
        def __init__(self, *sub_policies):
            self.sub_policies = sub_policies

        def __call__(self, img: Image):
            return random.choice(self.sub_policies)(img)


    @Register.augmentation_set(on_single=True, on_images=True)
    class AACifar10Aug(AbstractAug):
        """
        AutoAugment CIFAR policies
        """

        @classmethod
        def _get_train_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
            assert ds.raw_data_shape.num_dims() == 3
            default = dict(fill_color=(128, 128, 128))
            all_transforms = [
                transforms.RandomCrop(32, padding=4, fill=128),
                transforms.RandomHorizontalFlip(),
                RandomSubPolicies(
                    SubPolicy(Invert(p=0.1, m=7, **default), Contrast(p=0.2, m=6, **default)),
                    SubPolicy(Rotate(p=0.7, m=2, **default), TranslateX(p=0.3, m=9, **default)),
                    SubPolicy(Sharpness(p=0.8, m=1, **default), Sharpness(p=0.9, m=3, **default)),
                    SubPolicy(ShearY(p=0.5, m=8, **default), TranslateY(p=0.7, m=9, **default)),
                    SubPolicy(AutoContrast(p=0.5, m=8, **default), Equalize(p=0.9, m=2, **default)),

                    SubPolicy(ShearY(p=0.2, m=7, **default), Posterize(p=0.3, m=7, **default)),
                    SubPolicy(Color(p=0.4, m=3, **default), Brightness(p=0.6, m=7, **default)),
                    SubPolicy(Sharpness(p=0.3, m=9, **default), Brightness(p=0.7, m=9, **default)),
                    SubPolicy(Equalize(p=0.6, m=5, **default), Equalize(p=0.5, m=1, **default)),
                    SubPolicy(Contrast(p=0.6, m=7, **default), Sharpness(p=0.6, m=5, **default)),

                    SubPolicy(Color(p=0.7, m=7, **default), TranslateX(p=0.5, m=8, **default)),
                    SubPolicy(Equalize(p=0.3, m=7, **default), AutoContrast(p=0.4, m=8, **default)),
                    SubPolicy(TranslateY(p=0.4, m=3, **default), Sharpness(p=0.2, m=6, **default)),
                    SubPolicy(Brightness(p=0.9, m=6, **default), Color(p=0.2, m=8, **default)),
                    SubPolicy(Solarize(p=0.5, m=2, **default), Invert(p=0.0, m=3, **default)),

                    SubPolicy(Equalize(p=0.2, m=0, **default), AutoContrast(p=0.6, m=0, **default)),
                    SubPolicy(Equalize(p=0.2, m=8, **default), Equalize(p=0.6, m=4, **default)),
                    SubPolicy(Color(p=0.9, m=9, **default), Equalize(p=0.6, m=6, **default)),
                    SubPolicy(AutoContrast(p=0.8, m=4, **default), Solarize(p=0.2, m=8, **default)),
                    SubPolicy(Brightness(p=0.1, m=3, **default), Color(p=0.7, m=0, **default)),

                    SubPolicy(Solarize(p=0.4, m=5, **default), AutoContrast(p=0.9, m=3, **default)),
                    SubPolicy(TranslateY(p=0.9, m=9, **default), TranslateY(p=0.7, m=9, **default)),
                    SubPolicy(AutoContrast(p=0.9, m=2, **default), Solarize(p=0.8, m=3, **default)),
                    SubPolicy(Equalize(p=0.8, m=8, **default), Invert(p=0.1, m=3, **default)),
                    SubPolicy(TranslateY(p=0.7, m=9, **default), AutoContrast(p=0.9, m=1, **default)),
                ),
            ]
            return AugType.ON_RAW, all_transforms

        @classmethod
        def _get_test_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
            assert ds.raw_data_shape.num_dims() == 3
            return AugType.NONE, []


    @Register.augmentation_set(on_single=True, on_images=True)
    class AAImagenetAug(AbstractAug):
        """
        AutoAugment ImageNet policies
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
            default = dict(fill_color=(128, 128, 128))
            all_transforms = [
                transforms.RandomResizedCrop(crop_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                RandomSubPolicies(
                    SubPolicy(Posterize(p=0.4, m=8, **default), Rotate(p=0.6, m=9, **default)),
                    SubPolicy(Solarize(p=0.6, m=5, **default), AutoContrast(p=0.6, m=5, **default)),
                    SubPolicy(Equalize(p=0.8, m=8, **default), Equalize(p=0.6, m=3, **default)),
                    SubPolicy(Posterize(p=0.6, m=7, **default), Posterize(p=0.6, m=6, **default)),
                    SubPolicy(Equalize(p=0.4, m=7, **default), Solarize(p=0.2, m=4, **default)),

                    SubPolicy(Equalize(p=0.4, m=4, **default), Rotate(p=0.8, m=8, **default)),
                    SubPolicy(Solarize(p=0.6, m=3, **default), Equalize(p=0.6, m=7, **default)),
                    SubPolicy(Posterize(p=0.8, m=5, **default), Equalize(p=1.0, m=2, **default)),
                    SubPolicy(Rotate(p=0.2, m=3, **default), Solarize(p=0.6, m=8, **default)),
                    SubPolicy(Equalize(p=0.6, m=8, **default), Posterize(p=0.4, m=6, **default)),

                    SubPolicy(Rotate(p=0.8, m=8, **default), Color(p=0.4, m=0, **default)),
                    SubPolicy(Rotate(p=0.4, m=9, **default), Equalize(p=0.6, m=2, **default)),
                    SubPolicy(Equalize(p=0.0, m=7, **default), Equalize(p=0.8, m=8, **default)),
                    SubPolicy(Invert(p=0.6, m=4, **default), Equalize(p=1.0, m=8, **default)),
                    SubPolicy(Color(p=0.6, m=4, **default), Contrast(p=1.0, m=8, **default)),

                    SubPolicy(Rotate(p=0.8, m=8, **default), Color(p=1.0, m=2, **default)),
                    SubPolicy(Color(p=0.8, m=8, **default), Solarize(p=0.8, m=7, **default)),
                    SubPolicy(Sharpness(p=0.4, m=7, **default), Invert(p=0.6, m=8, **default)),
                    SubPolicy(ShearX(p=0.6, m=5, **default), Equalize(p=1.0, m=9, **default)),
                    SubPolicy(Color(p=0.4, m=0, **default), Equalize(p=0.6, m=3, **default)),

                    SubPolicy(Equalize(p=0.4, m=7, **default), Solarize(p=0.2, m=4, **default)),
                    SubPolicy(Solarize(p=0.6, m=5, **default), AutoContrast(p=0.6, m=5, **default)),
                    SubPolicy(Invert(p=0.6, m=4, **default), Equalize(p=1.0, m=8, **default)),
                    SubPolicy(Color(p=0.6, m=4, **default), Contrast(p=1.0, m=8, **default)),
                    SubPolicy(Equalize(p=0.8, m=8, **default), Equalize(p=0.6, m=3, **default)),
                ),
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

