import torch
from uninas.data.abstract import AbstractDataSet, AbstractAug, BatchAugmentations, AbstractBatchAugmentation
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


class MixUp(AbstractBatchAugmentation):
    def __init__(self, num_classes: int, alpha1=1.0, alpha2=1.0):
        self.num_classes = num_classes
        self.distribution = torch.distributions.beta.Beta(alpha1, alpha2)

    def __call__(self, data: torch.Tensor, labels: torch.Tensor):
        indices = torch.randperm(data.size(0)).to(data.device)
        labels_oh1 = torch.nn.functional.one_hot(labels, self.num_classes)
        labels_oh2 = torch.nn.functional.one_hot(labels[indices], self.num_classes)
        data2 = data[indices]
        lambda_ = self.distribution.sample(sample_shape=(1,)).to(data.device)
        mixed_data = data * lambda_ + data2 * (1 - lambda_)
        mixed_labels = labels_oh1 * lambda_ + labels_oh2 * (1 - lambda_)
        return mixed_data, mixed_labels


@Register.augmentation_set(on_batch=True, on_images=True)
class MixUpAug(AbstractAug):
    """
    mixup: Beyond Empirical Risk Minimization
    https://arxiv.org/abs/1710.09412

    each data point in the batch is assigned a random partner, interpolates data and label between the two
    (the batch size stays the same, A partnering with B does not mean B partnering with A)
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('alpha1', default=1.0, type=float, help='first weight for the beta distribution'),
            Argument('alpha2', default=1.0, type=float, help='second weight for the beta distribution'),
        ]

    @classmethod
    def _get_train_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (list, [BatchAugmentations]):
        alpha1, alpha2 = cls._parsed_arguments(['alpha1', 'alpha2'], args, index=index)
        return [], [MixUp(ds.num_classes(), alpha1, alpha2)]

    @classmethod
    def _get_test_transforms(cls, args: Namespace, index: int, ds: AbstractDataSet) -> (list, [BatchAugmentations]):
        return [], []
