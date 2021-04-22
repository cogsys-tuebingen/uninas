"""
Improved Regularization of Convolutional Neural Networks with Cutout
https://arxiv.org/abs/1708.04552
"""

import numpy as np
from uninas.data.abstract import AbstractDataSet, AbstractAug, AugType
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


class Cutout(object):

    def __init__(self, length: int):
        self.length = length

    def __call__(self, img):
        w, h = img.width, img.height
        mask = np.ones((h, w), np.uint8)
        y = np.random.randint(h)
        x = np.random.randint(w)

        l2 = self.length // 2
        y1 = np.clip(y - l2, 0, h)
        y2 = np.clip(y + l2, 0, h)
        x1 = np.clip(x - l2, 0, w)
        x2 = np.clip(x + l2, 0, w)

        mask[y1: y2, x1: x2] = 0
        return np.array(img.getdata(), np.uint8).reshape((h, w, 3)) * np.expand_dims(mask, axis=-1)


@Register.augmentation_set(on_single=True, on_images=True)
class CutoutAug(AbstractAug):
    """
    Zeroes a random patch of the image, enforcing the networks to avoid looking for only one kind of feature
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('size', default=16, type=int, help='enable cutout, squares of given length'),
        ]

    @classmethod
    def _get_train_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
        assert ds.raw_data_shape.num_dims() == 3
        size = cls._parsed_argument('size', args, index=index)
        if size > 0:
            return AugType.ON_RAW, [Cutout(size)]
        return AugType.NONE, []

    @classmethod
    def _get_test_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
        assert ds.raw_data_shape.num_dims() == 3
        return AugType.NONE, []
