from uninas.data.abstract import AbstractDataSet, AbstractAug, AugType
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


try:
    import albumentations as alb


    class AbstractAlbAug(AbstractAug):
        _apply_to_test = False
        _alb_cls = None
        _alb_defaults = dict()

        @classmethod
        def applies_to_labels(cls) -> bool:
            return True

        @classmethod
        def _get_train_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
            assert ds.raw_data_shape.num_dims() == 3
            all_parsed = cls._all_parsed_arguments(args, index=index)
            return AugType.ON_RAW, [cls._alb_cls(**all_parsed, **cls._alb_defaults)]

        @classmethod
        def _get_test_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (AugType, list):
            if cls._apply_to_test:
                return cls._get_train_transforms(args, index, ds)
            return AugType.NONE, []


    @Register.augmentation_set(on_single=True, on_images=True)
    class PadIfNeededAlbAug(AbstractAlbAug):
        _apply_to_test = True
        _alb_cls = alb.PadIfNeeded
        _alb_defaults = dict()

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return super().args_to_add(index) + [
                Argument('min_height', default=300, type=int, help='to which size to pad the images'),
                Argument('min_width', default=300, type=int, help='to which size to pad the images'),
                Argument('value', default=0, type=int, help='with which value to pad the images'),
            ]


    @Register.augmentation_set(on_single=True, on_images=True)
    class HorizontalFlipAlbAug(AbstractAlbAug):
        _apply_to_test = False
        _alb_cls = alb.HorizontalFlip
        _alb_defaults = dict()

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return super().args_to_add(index) + [
                Argument('p', default=0.5, type=float, help='probability to apply'),
            ]


    @Register.augmentation_set(on_single=True, on_images=True)
    class VerticalFlipAlbAug(AbstractAlbAug):
        _apply_to_test = False
        _alb_cls = alb.VerticalFlip
        _alb_defaults = dict()

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return super().args_to_add(index) + [
                Argument('p', default=0.5, type=float, help='probability to apply'),
            ]


    @Register.augmentation_set(on_single=True, on_images=True)
    class RandomRotate90AlbAug(AbstractAlbAug):
        _apply_to_test = False
        _alb_cls = alb.RandomRotate90
        _alb_defaults = dict(k=1)

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return super().args_to_add(index) + [
                Argument('p', default=0.5, type=float, help='probability to apply'),
            ]


except ImportError as e:
    Register.missing_import(e)
