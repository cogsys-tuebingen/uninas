from uninas.utils.correlations.pearson import PearsonCorrelation
from uninas.register import Register


try:
    from scipy.stats import spearmanr


    @Register.correlation_metric(rank=True)
    class SpearmanCorrelation(PearsonCorrelation):
        """
        Plot scattered data and calculate a correlation value
        Spearman's rank correlation coefficient: https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
        """

        @classmethod
        def short_name(cls) -> str:
            return "SCC"

        @classmethod
        def calculate(cls, data0: list, data1: list) -> float:
            """
            calculate and return the correlation value
            """
            r, p = spearmanr(data0, data1)
            return r


    if __name__ == '__main__':
        import random
        scc = SpearmanCorrelation(column_names=('predicted accuracy', 'true accuracy'), add_lines=True, can_show=True)
        x = [v/10 for v in range(-10, 10, 1)]
        y1 = [xi**3 for xi in x]
        y2 = [xi**4 for xi in x]
        y3 = [0 for xi in x]
        x2 = [(v+random.random())*0.01 for v in range(-10, 10, 1)]
        y21 = [xi + (random.random()-0.5) for xi in x]
        scc.add_data(x, y1, 'data #1', other_metrics=(PearsonCorrelation,))
        scc.add_data(x, y2, 'data #2', other_metrics=(PearsonCorrelation,))
        scc.add_data(x, y3, 'data #3', other_metrics=(PearsonCorrelation,))
        scc.add_data(x2, y21, 'data #4', other_metrics=(PearsonCorrelation,))
        scc.plot(title=scc.__class__.__name__, legend=True, show=True, save_path=None)

except ImportError as e:
    Register.missing_import(e)
