"""
Plot scattered data
Kendall Tau correlation: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
"""

import os
import numpy as np


class KendallTau:
    def __init__(self, column_names=('A', 'B'), add_lines=True, can_show=True):
        self.column_names = column_names
        self.add_lines = add_lines
        self.can_show = can_show
        import matplotlib
        if not self.can_show:
            matplotlib.use('pdf')
        import matplotlib.pyplot as plt
        self.ax = plt.axes()

    def add_data(self, data0: list, data1: list, label: str, plot=True, **plot_kwargs) -> float:
        ds = sorted([(i, d0) for i, d0 in enumerate(data0)], key=lambda d: d[1])
        data0 = [d[1] for d in ds]
        data1 = [data1[d[0]] for d in ds]

        concordant, discordant = 0, 0
        for i in range(len(data0)-1):
            for j in range(i+1, len(data0)):
                if (data0[i] < data0[j] and data1[i] < data1[j]) or (data0[i] > data0[j] and data1[i] > data1[j]):
                    concordant += 1
                elif (data0[i] > data0[j] and data1[i] < data1[j]) or (data0[i] < data0[j] and data1[i] > data1[j]):
                    discordant += 1
        n = len(data0)
        tau = (concordant - discordant) / ((n * (n-1)) // 2)
        if plot:
            self.ax.scatter(data0, data1, label='%s, KT: %.2f' % (label, tau), **plot_kwargs)
            if self.add_lines:
                poly = np.poly1d(np.polyfit(data0, data1, 1))
                y = [poly(d0) for d0 in data0]
                self.ax.plot(data0, y)
        return tau

    def plot(self, title='', legend=True, show=True, save_path: str = None, **plot_kwargs):
        import matplotlib.pyplot as plt
        plt.xlabel(self.column_names[0])
        plt.ylabel(self.column_names[1])
        if len(title) > 0:
            plt.title(title)
        if legend:
            self.ax.legend()
        plt.plot(**plot_kwargs)
        if show and self.can_show:
            plt.show()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    kt = KendallTau(column_names=('predicted accuracy', 'true accuracy'), add_lines=True, can_show=True)
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
    kt.plot(title='kendall tau', legend=True, show=True, save_path=None)
