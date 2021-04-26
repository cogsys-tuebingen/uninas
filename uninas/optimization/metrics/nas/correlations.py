import os
import numpy as np
import matplotlib.pyplot as plt
from uninas.optimization.metrics.nas.abstract import AbstractNasMetric
from uninas.register import Register


try:
    from scipy.stats import kendalltau, pearsonr, spearmanr


    class AbstractCorrelationNasMetric(AbstractNasMetric):
        """
        Calculate metrics based on predicted/true network performance values,
        how the ranking correlation changes,
        when we consider fewer networks as selected by a NAS algorithm
        """

        _short_name = None
        _scipy_fun = None
        _x_label = "top n networks"

        @classmethod
        def plot_correlations(cls, predictions: np.array, targets: np.array, corr_classes: list,
                              axes_names=('predictions', 'targets'), show=True, save_path: str = None):
            """
            :param predictions: list of target predictions
            :param targets: list of targets
            :param corr_classes: list of AbstractCorrelationNasMetric classes that are to be evaluated
            :param axes_names: labels for the axes
            :param show: whether to show
            :param save_path: path to save, do not save if None
            """
            texts = []
            for corr_cls in corr_classes:
                assert issubclass(corr_cls, AbstractCorrelationNasMetric)
                texts.append("%s: %.2f" % (corr_cls.short_name(), corr_cls.correlation_value(predictions, targets)))
            plt.scatter(predictions, targets, label=", ".join(texts))
            plt.xlabel(axes_names[0])
            plt.ylabel(axes_names[1])
            plt.legend()
            if show:
                plt.show()
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
            plt.close()
            plt.cla()

        @classmethod
        def correlation_value(cls, predictions: np.array, targets: np.array) -> float:
            r, _ = cls._scipy_fun(predictions, targets)
            return r

        @classmethod
        def _get_data(cls, predictions: np.array, targets: np.array) -> {str: np.array}:
            """
            :param predictions: network predictions (e.g. accuracy)
            :param targets: ground truth values
            """
            predictions, targets = cls._sorted_by_targets(predictions, targets, ascending=True)
            kt = []
            nm = len(predictions)
            for i in range(nm):
                try:
                    kt.append(cls.correlation_value(predictions[i:], targets[i:]))
                except:
                    kt.append(np.nan)
            return dict(corr=np.array(kt))

        @classmethod
        def _plot_to_axis(cls, ax: plt.Axes, x: np.array, data: {str: np.array}, name: str, index=0, has_multiple=False,
                          last_on_axis=True, rem_last=1, prev_state={}, **_) -> dict:
            ax.set_ylabel('%s, correlation' % cls.short_name())
            mean = np.mean(data.get('corr'), axis=0)
            std = np.std(data.get('corr'), axis=0)
            label = "%s, mean + std" if data.get('corr').shape[0] > 1 else "%s, mean"
            ax.plot(x, mean, cls._markers[index], label=label % name, color=cls._cols[index])
            ax.fill_between(x, mean - std, mean + std, alpha=0.3, color=cls._cols[index])

            cls._update_state_mean(prev_state, mean, rem_last=rem_last)
            cls._limit_ax_by_mean(prev_state, ax, last_on_axis=last_on_axis, min_y=-1, max_y=1, mul=1.1)
            return prev_state


    @Register.nas_metric(is_correlation=True)
    class KendallTauNasMetric(AbstractCorrelationNasMetric):
        """
        Calculate metrics based on predicted/true network performance values,
        how the ranking correlation changes,
        when we consider fewer networks as selected by a NAS algorithm

        Kendall Tau correlation: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
        """

        _short_name = "KT"
        _scipy_fun = kendalltau


    @Register.nas_metric(is_correlation=True)
    class PearsonNasMetric(AbstractCorrelationNasMetric):
        """
        Calculate metrics based on predicted/true network performance values,
        how the ranking correlation changes,
        when we consider fewer networks as selected by a NAS algorithm

        Pearson correlation coefficient: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        """

        _short_name = "PCC"
        _scipy_fun = pearsonr


    @Register.nas_metric(is_correlation=True)
    class SpearmanNasMetric(AbstractCorrelationNasMetric):
        """
        Calculate metrics based on predicted/true network performance values,
        how the ranking correlation changes,
        when we consider fewer networks as selected by a NAS algorithm

        Spearman's rank correlation coefficient: https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
        """

        _short_name = "SCC"
        _scipy_fun = spearmanr


    if __name__ == '__main__':
        d0, d1 = np.random.multivariate_normal([0.5, 0.5], [[1, 3], [0, 1]], size=1000).T
        metric = KendallTauNasMetric()
        dx = metric.get_data(d0, d1)
        metric.plot(dx, show=True)


except ImportError as e:
    Register.missing_import(e)
