from typing import Union, List, Dict
from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
from uninas.register import Register


class AbstractNasMetric:
    """
    Calculate metrics based on predicted/true network performance values
    """
    _cols = ['red', 'green', 'blue', 'orange', 'cyan', 'purple', 'gray', 'deepskyblue', 'gold']
    _markers = ['o--', 'v--', '^--', '<--', '>--', '*--', 'X--', 'd--', '--']

    _short_name = None
    _x_label = None

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @classmethod
    def _sorted_by_predictions(cls, predictions: np.array, targets: np.array, ascending=True) -> (np.array, np.array):
        """
        sorts predictions and targets by the predictions
        :param predictions: network predictions (e.g. accuracy)
        :param targets: ground truth values
        """
        idx = np.argsort(predictions)
        if not ascending:
            idx = list(reversed(idx))
        return predictions[idx], targets[idx]

    @classmethod
    def _sorted_by_targets(cls, predictions: np.array, targets: np.array, ascending=True) -> (np.array, np.array):
        """
        sorts predictions and targets by the targets
        :param predictions: network predictions (e.g. accuracy)
        :param targets: ground truth values
        """
        idx = np.argsort(targets)
        if not ascending:
            idx = list(reversed(idx))
        return predictions[idx], targets[idx]

    @classmethod
    def get_data(cls, predictions: Union[list, np.array], targets: Union[list, np.array]) -> {str: np.array}:
        """
        :param predictions: network predictions (e.g. accuracy)
        :param targets: ground truth values
        """
        assert len(predictions) == len(targets)
        predictions = np.array(predictions)
        targets = np.array(targets)
        return cls._get_data(predictions, targets)

    @classmethod
    def _get_data(cls, predictions: np.array, targets: np.array) -> {str: np.array}:
        """
        :param predictions: network predictions (e.g. accuracy)
        :param targets: ground truth values
        """
        raise NotImplementedError

    @classmethod
    def _update_state_mean(cls, state: dict, mean: np.array, rem_last=1):
        """ keep track of min/max values that will be plotted on the y axis """
        mean = mean[:-rem_last]
        if len(mean) > 0:
            state['_mean_y_min'] = min([state.get('_mean_y_min', 99999999), min(mean)])
            state['_mean_y_max'] = max([state.get('_mean_y_max', -99999999), max(mean)])

    @classmethod
    def _limit_ax_by_mean(cls, state: dict, ax: plt.axis, last_on_axis=True,
                          min_y: float = None, max_y: float = None, mul=1.1):
        """ set y limits for the axis """
        if last_on_axis and ('_mean_y_min' in state):
            min_ = state['_mean_y_min'] - (mul - 1.0) * abs(state['_mean_y_min'])
            max_ = state['_mean_y_max'] + (mul - 1.0) * abs(state['_mean_y_max'])
            if isinstance(min_y, float):
                min_ = max([min_, min_y])
            if isinstance(max_y, float):
                max_ = min([max_, max_y])
            ax.set_ylim(bottom=min_, top=max_)

    @classmethod
    def plot(cls, data: {str: [float]}, title='', legend=True, log_x=False, xlim=1, show=True, save_path: str = None):
        ax = plt.axes(label="%s_%s" % (cls.__name__, str(id(data))))
        if len(title) > 0:
            plt.title(title)
        cls.plot_to_axis(ax, data, "", legend=legend, log_x=log_x, xlim=xlim)
        if show:
            plt.show()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close()
        plt.cla()

    @classmethod
    def plot_multiple(cls, data_by_cls: {str: {str: Union[dict, list]}},
                      title='', legend=True, log_x=False, xlim=1,
                      show=True, save_path: str = None):
        """
        plots the data of multiple classes to an axis each
        :param data_by_cls: data to plot
                            {group name: {metric name: [metric.get_data() outputs]}}
        :param title: plot title
        :param legend: whether to add legends
        :param log_x: whether to represent the x axis in log scale
        :param xlim: limit where to stop plotting the best predictions/targets
        :param show: whether to show the plot
        :param save_path: where to save, not saved if None
        :return:
        """
        # create axes, set title
        data_names = list(data_by_cls.keys())
        data_keys = list(data_by_cls[data_names[0]].keys())
        fig, axes = plt.subplots(len(data_keys), 1)
        if len(data_keys) <= 1:
            axes = [axes]
        if len(title) > 0:
            axes[0].set_title(title)
        # check where to add xlabel and xticks
        add_labels = [True for _ in range(len(axes))]
        tick_labels = [Register.nas_metrics.get(cls_).get_xlabel() for cls_ in data_by_cls[data_names[0]].keys()]
        for i in range(len(tick_labels) - 1):
            if tick_labels[i] == tick_labels[i+1]:
                add_labels[i] = False
        # plot data to axes
        for ax, key, add in zip(axes, data_keys, add_labels):
            cls_ = Register.nas_metrics.get(key)
            prev_state = {}
            for i, name in enumerate(data_names):
                data = data_by_cls.get(name).get(key, {})
                prev_state = cls_.plot_to_axis(ax, data, name, legend=legend,
                                               xlabel=add, xtick_labels=add, log_x=log_x, xlim=xlim,
                                               index=i, has_multiple=len(data_names) > 1,
                                               first_on_axis=i == 0, last_on_axis=i == len(data_names)-1,
                                               prev_state=prev_state)
            ax.grid()
        # plt.tight_layout(h_pad=0.1)
        if show:
            plt.show()
        if isinstance(save_path, str):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            # print("saved plot to %s" % save_path)
        plt.close()

    @classmethod
    def plot_to_axis(cls, ax: plt.Axes, data: Union[Dict[str, np.array], List[Dict[str, np.array]]], name: str,
                     legend=True, xlabel=True, xtick_labels=True, log_x=True, xlim=1, **plot_kwargs) -> dict:
        """
        plots the data to an axis
        :return: dict of plotting state
        """
        # merge multiple runs into one dict
        data_merged = defaultdict(list)
        if isinstance(data, dict):
            data = [data]
        for dx in data:
            for dk, dv in dx.items():
                data_merged[dk].append(dv)
        data_merged = {k: np.stack(v, axis=0) for k, v in data_merged.items()}
        # x axis data
        x = None
        for v in data_merged.values():
            x = list(reversed(range(1, len(v[0]) + 1)))
            break
        # plot
        r = cls._plot_to_axis(ax, x, data_merged, name, rem_last=max([xlim-1, 1]), **plot_kwargs)
        ax.grid()
        ax.set_xlim(max(x), xlim)
        if log_x:
            ax.set_xscale('log')
        if xlabel:
            ax.set_xlabel(cls.get_xlabel())
        if not xtick_labels:
            ax.xaxis.set_ticklabels([])
        if legend:
            ax.legend()
        return r

    @classmethod
    def _plot_to_axis(cls, ax: plt.Axes, x: np.array, data: {str: np.array}, name: str,
                      has_multiple=True, index=0, first_on_axis=True, last_on_axis=True, rem_last=1, prev_state={},
                      **_) -> dict:
        """
        plots the data to an axis
        :param ax: plt axis to plot to
        :param data: {key: np.array(runs, data)} as returned from get_data,
                     but possibly containing data of multiple runs
        :param name: name in the legend
        :param has_multiple: whether multiple function calls will be made to plot
        :param index: index of this data in the to-be-plotted order to this axis
        :param first_on_axis: whether this data is the first to be plotted to the axis
        :param last_on_axis: whether this data is the last to be plotted to the axis
        :param rem_last: how many data values will not be plotted (due to setting axis.xlim)
        :param prev_state: plotting state that the previous function call returned
        :return: dict of plotting state
        """
        raise NotImplementedError

    @classmethod
    def short_name(cls) -> str:
        return cls._short_name

    @classmethod
    def get_xlabel(cls) -> str:
        return cls._x_label
