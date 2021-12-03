import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from uninas.data.abstract import AbstractDataSet
from uninas.models.networks.abstract import AbstractNetwork
from uninas.training.metrics.abstract import AbstractAccumulateMetric, ResultValue
from uninas.utils.args import Namespace, Argument
from uninas.register import Register


def save_confusion_matrix(save_path: str, cm: np.array, norm_by_columns=True, classes: [str] = None,
                          title: str = None, cmap=plt.get_cmap("summer")):
    """
    plot and save a confusion matrix
    """
    if classes is None:
        classes = [str(i) for i in range(len(cm))]

    # plot a normalized confusion matrix
    sums = np.sum(cm, axis=0 if norm_by_columns else 1)
    sums = np.expand_dims(sums, axis=0 if norm_by_columns else 1)
    sums[sums == 0] = 1
    cm_norm = np.divide(cm, sums)
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_xticks(np.arange(len(classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(classes) + 1) - .5, minor=True)
    plt.setp(ax.get_xticklabels(), rotation=-40, ha="right", rotation_mode="anchor")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.figure.colorbar(im, ax=ax)

    ax.set_title(title if isinstance(title, str) else "Confusion matrix (output \\ target)")
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)


@Register.metric(only_head=True)
class ConfusionMatrixMetric(AbstractAccumulateMetric):
    """
    Count the class predictions vs ground truth in a NxN matrix
    """

    def get_log_name(self) -> str:
        return 'confusion_matrix'

    @classmethod
    def from_args(cls, args: Namespace, index: int, data_set: AbstractDataSet, head_weights: list) -> 'ConfusionMatrixMetric':
        """
        :param args: global arguments namespace
        :param index: index of this metric
        :param data_set: data set that is evaluated on
        :param head_weights: how each head is weighted
        """
        assert data_set.is_classification(), "Accuracy can only be evaluated on classification data"
        num_classes = data_set.num_classes()
        class_names = data_set.get_class_names()
        all_parsed = cls._all_parsed_arguments(args, index=index)
        return cls(head_weights=head_weights, num_classes=num_classes, class_names=class_names, **all_parsed)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('ignore_target_index', default=-999, type=int, help='remove all samples where the target matches this index.'),
            Argument('ignore_prediction_index', default=-999, type=int, help='if the network predicts this index, choose the next most-likely prediction instead.'),
        ]

    def _viz_stats(self, save_path: str, stats: dict):
        """
        visualize this metric

        :param save_path: where to save
        :param stats: accumulated stats throughout the _evaluate() calls, after calling _update_stats() on them
        :return:
        """
        matrix = stats['matrix'].detach().cpu().numpy()
        save_confusion_matrix("%s_cm.pdf" % save_path, matrix, classes=self.class_names)

    def _evaluate(self, net: AbstractNetwork, inputs: torch.Tensor,
                  logits: [torch.Tensor], targets: torch.Tensor) -> {str: ResultValue}:
        """

        :param net: evaluated network
        :param inputs: network inputs
        :param logits: network outputs
        :param targets: output targets
        :return: dictionary of string keys with corresponding results
        """
        logits, targets = self._batchify_tensors([logits[-1]], targets)
        targets = self._remove_onehot(targets)

        logits, targets = self._ignore_with_index(logits, targets, self.ignore_target_index, self.ignore_prediction_index)
        logits = self._remove_onehot(logits[0])

        matrix = torch.zeros(size=(self.num_classes, self.num_classes), dtype=torch.int32, device=logits.device)
        for idx_p, idx_t in zip(logits, targets):
            matrix[idx_p, idx_t] += 1
        return {'matrix': ResultValue(matrix, logits.shape[0])}


if __name__ == '__main__':
    matrix_ = np.array(
        [[20, 10, 1],
         [5, 5, 1],
         [3, 11, 19]]
    )
    save_confusion_matrix('/tmp/uninas/viz/cm.pdf', matrix_, norm_by_columns=True)
