from scipy.stats import kendalltau
from uninas.utils.correlations.abstract import AbstractCorrelation
from uninas.register import Register


@Register.correlation_metric(rank=True)
class KendallTauCorrelation(AbstractCorrelation):
    """
    Plot scattered data and calculate a correlation value
    Kendall Tau correlation: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    """

    @classmethod
    def short_name(cls) -> str:
        return "KT"

    @classmethod
    def calculate(cls, data0: list, data1: list) -> float:
        """
        calculate and return the correlation value
        """
        r, p = kendalltau(data0, data1)
        return r


if __name__ == '__main__':
    kt = KendallTauCorrelation(column_names=('predicted accuracy', 'true accuracy'), add_lines=True, can_show=True)
    kt.add_data([0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49],
                [0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0.70],
                'data #1')
    kt.add_data([0.50, 0.53, 0.52, 0.56, 0.55, 0.51, 0.58, 0.59, 0.54, 0.57],
                [0.70, 0.73, 0.72, 0.76, 0.75, 0.71, 0.78, 0.79, 0.74, 0.77],
                'data #2')
    kt.add_data([0.60, 0.62, 0.61, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69],
                [0.70, 0.73, 0.72, 0.71, 0.75, 0.76, 0.74, 0.77, 0.78, 0.79],
                'data #3')
    kt.add_data([0.73, 0.70, 0.71, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79],
                [0.70, 0.73, 0.71, 0.75, 0.77, 0.72, 0.74, 0.76, 0.79, 0.78],
                'data #4')
    kt.add_data([0.83, 0.80, 0.81, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89],
                [0.77, 0.75, 0.78, 0.73, 0.79, 0.72, 0.74, 0.76, 0.71, 0.70],
                'data #5')
    kt.plot(title=kt.__class__.__name__, legend=True, show=True, save_path=None)