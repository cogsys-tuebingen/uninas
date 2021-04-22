import numpy as np
import matplotlib.pyplot as plt
from uninas.optimization.metrics.nas.abstract import AbstractNasMetric
from uninas.register import Register


class ByXNasMetric(AbstractNasMetric):
    """
    Calculate metrics based on predicted/true network performance values,
    how much the network quality improves over the worst possible one,
    when we consider fewer networks as selected by a NAS algorithm
    """

    _short_name = None
    _x_label = None
    _y_label = None

    @classmethod
    def _get_data(cls, predictions: np.array, targets: np.array) -> {str: np.array}:
        """
        :param predictions: network predictions (e.g. accuracy)
        :param targets: ground truth values
        """
        raise NotImplementedError

    @classmethod
    def _plot_to_axis(cls, ax: plt.Axes, x: np.array, data: {str: np.array}, name: str, has_multiple=True, index=0,
                      **_) -> dict:
        """
        plots the data to an axis
        :param ax: plt axis to plot to
        :param data: {key: np.array(runs, data)} as returned from get_data,
                     but possibly containing data of multiple runs
        :param name: name
        :param has_multiple: whether multiple plots will be added to this axis
        :return: dict of plotting state
        """
        ax.set_ylabel(cls._y_label)
        if data.get('min').shape[0] == 1:
            mean_min = np.min(data.get('min'), axis=0)
            mean_max = np.max(data.get('max'), axis=0)
            ax.fill_between(x, mean_min, mean_max, alpha=0.15, color=cls._cols[index])
        for i, v in enumerate(data.get('values')):
            ax.scatter(x, v, label=("%s, all networks" % name) if i == 0 else None, color=cls._cols[index], s=2)
        return {}


@Register.nas_metric()
class ByPredictionNasMetric(ByXNasMetric):
    """
    Calculate metrics based on predicted/true network performance values,
    how much the network quality improves over the worst possible one,
    when we consider fewer networks as selected by a NAS algorithm
    """

    _short_name = "by prediction"
    _x_label = "top n best predicted networks"
    _y_label = "ground truth"

    @classmethod
    def _get_data(cls, predictions: np.array, targets: np.array) -> {str: np.array}:
        """
        :param predictions: network predictions (e.g. accuracy)
        :param targets: ground truth values
        """
        predictions, targets = cls._sorted_by_predictions(predictions, targets, ascending=True)
        min_, max_ = [], []
        for i in range(len(predictions)):
            min_.append(np.min(targets[i:]))
            max_.append(np.max(targets[i:]))
        return dict(min=np.array(min_), max=np.array(max_), values=np.array(targets))


@Register.nas_metric()
class ByTargetsNasMetric(ByXNasMetric):
    """
    Calculate metrics based on predicted/true network performance values,
    how much the network quality improves over the worst possible one,
    when we consider fewer networks as selected by a NAS algorithm
    """

    _short_name = "by prediction"
    _x_label = "top n networks"
    _y_label = "predictions"

    @classmethod
    def _get_data(cls, predictions: np.array, targets: np.array) -> {str: np.array}:
        """
        :param predictions: network predictions (e.g. accuracy)
        :param targets: ground truth values
        """
        predictions, targets = cls._sorted_by_targets(predictions, targets, ascending=True)
        min_, max_ = [], []
        for i in range(len(predictions)):
            min_.append(np.min(predictions[i:]))
            max_.append(np.max(predictions[i:]))
        return dict(min=np.array(min_), max=np.array(max_), values=np.array(predictions))
