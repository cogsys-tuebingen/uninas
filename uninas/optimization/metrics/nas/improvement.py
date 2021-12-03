import numpy as np
import matplotlib.pyplot as plt
from uninas.optimization.metrics.nas.abstract import AbstractNasMetric
from uninas.register import Register


@Register.nas_metric()
class ImprovementNasMetric(AbstractNasMetric):
    """
    Calculate metrics based on predicted/true network performance values,
    how much the network quality improves over the worst possible one,
    when we consider fewer networks as selected by a NAS algorithm.

    The improvement plotted absolute, but also normalized:
        1 is the best evaluated network
        0 is the average over all evaluated networks
        -1 is the worst evaluated network
    The scaling between [0, 1] and [-1, 0] may differ.
    """

    _short_name = "improvement"
    _x_label = "top n best predicted networks"

    @classmethod
    def _get_data(cls, predictions: np.array, targets: np.array) -> {str: np.array}:
        """
        :param predictions: network predictions (e.g. accuracy)
        :param targets: ground truth values
        """
        predictions, targets = cls._sorted_by_predictions(predictions, targets, ascending=True)
        mean_ = [np.mean(targets[i:]) for i in range(len(predictions))]
        global_mean, global_min, global_max = mean_[0], np.min(targets), np.max(targets)
        return dict(mean=mean_, global_mean=global_mean, global_min=global_min, global_max=global_max)

    @classmethod
    def _plot_to_axis(cls, ax: plt.Axes, x: np.array, data: {str: np.array}, name: str, has_multiple=True, index=0,
                      first_on_axis=True, last_on_axis=True, rem_last=1, prev_state={}, **_) -> dict:
        # get absolute min/mean/max values
        abs_mean = np.mean(data.get('global_mean'))
        abs_min = np.mean(data.get('global_min'))
        abs_max = np.mean(data.get('global_max'))
        mean_mean = np.mean(data.get('mean'), axis=0)

        # axis scaling, only relevant part on the y axis
        d = data.get('mean')
        if rem_last > 1:
            d = d[:, :-rem_last]
        y0 = np.mean(np.min(d, axis=1))
        y1 = np.mean(np.max(d, axis=1))

        # update abs values
        prev_state['abs_min'] = min([abs_min, prev_state.get('abs_min', abs_min)])
        prev_state['abs_max'] = max([abs_max, prev_state.get('abs_max', abs_max)])
        prev_state['abs_mean_values'] = prev_state.get('abs_mean_values', [])
        prev_state['abs_mean_values'].append(abs_mean)
        prev_state['y0'] = min([y0, prev_state.get('y0', y0)])
        prev_state['y1'] = max([y1, prev_state.get('y1', y1)])

        # axis scaling, only relevant part on the y axis
        if last_on_axis:
            ax.set_ylabel('mean ground truth accuracy')
            ax.set_ylim(prev_state['y0'], prev_state['y1'])
            ax2 = ax.twinx()
            ax2.set_ylabel('improvement')
            mean = np.mean(prev_state['abs_mean_values'])
            diff = prev_state['abs_max'] - mean
            y0_rel = (prev_state['y0'] - mean) / diff
            y1_rel = (prev_state['y1'] - mean) / diff
            ax2.set_ylim(y0_rel, y1_rel)

        label = "%s, mean + std" if data.get('mean').shape[0] > 1 else "%s, mean"
        ax.plot(x, mean_mean, cls._markers[index], label=label % name, color=cls._cols[index])
        if data.get('mean').shape[0] > 1:
            std_mean = np.std(data.get('mean'), axis=0)
            ax.fill_between(x, mean_mean - std_mean, mean_mean + std_mean, alpha=0.3, color=cls._cols[index])
            # print("name={:>20}, rem={} \t mean={:.2f}, std={:.2f}".format(name, rem_last, mean_mean[-rem_last], std_mean[-rem_last]))

        # cls._update_state_mean(prev_state, mean_mean)
        # cls._limit_ax_by_mean(prev_state, ax, last_on_axis=last_on_axis, mul=1.1)
        return prev_state


if __name__ == '__main__':
    from uninas.builder import Builder
    from uninas.optimization.metrics.nas.by_value import ByTargetsNasMetric
    from uninas.optimization.metrics.nas.correlations import KendallTauNasMetric, PearsonNasMetric
    Builder()

    jobs = ['1', '2']
    data_ = []
    for _ in jobs:
        ds = [np.random.multivariate_normal([0, 0], [[1, 4], [0, 1]], size=1000).T for _ in range(3)]
        data_.append({
            ImprovementNasMetric.__name__: [ImprovementNasMetric.get_data(d0, d1) for d0, d1 in ds],
            # ByTargetsNasMetric.__name__: [ByTargetsNasMetric.get_data(d0, d1) for d0, d1 in ds],
            # KendallTauNasMetric.__name__: [KendallTauNasMetric.get_data(d0, d1) for d0, d1 in ds],
            # PearsonNasMetric.__name__: [PearsonNasMetric.get_data(d0, d1) for d0, d1 in ds],
        })
    data_ = {j: d for j, d in zip(jobs, data_)}
    ImprovementNasMetric.plot_multiple(data_, show=True, log_x=True, xlim=10)
