import os
from typing import Iterable
import numpy as np


class AbstractCorrelation:
    """
    Plot scattered data and calculate a correlation value
    """

    def __init__(self, column_names=('A', 'B'), add_lines=True, can_show=True):
        self.column_names = column_names
        self.add_lines = add_lines
        self.can_show = can_show
        import matplotlib
        if not self.can_show:
            matplotlib.use('pdf')
        import matplotlib.pyplot as plt
        self.ax = plt.axes()

    def add_data(self, data0: list, data1: list, label: str, other_metrics: tuple = (), plot=True, **plot_kwargs) -> float:
        """
        add data to the plot (if wanted), calculate and return the correlation value
        :param data0:
        :param data1:
        :param label: label for the data
        :param other_metrics: tuple of other metric-classes to also add
        :param plot: whether to generate a plot or not
        :param plot_kwargs: kwargs for the plt scatter function
        """
        if plot:
            # calculate all metrics, generate label
            r = None
            all_labels = []
            for m in other_metrics:
                r_ = m.calculate(data0, data1)
                all_labels.append((m.short_name(), r_))
                if self.__class__ == m:
                    r = r_
            if r is None:
                r = self.calculate(data0, data1)
                all_labels.append((self.short_name(), r))
            plt_label = "{}, {}".format(label, ", ".join(["%s: %.2f" % (v0, v1) for (v0, v1) in all_labels]))

            # plot
            self.ax.scatter(data0, data1, label=plt_label, **plot_kwargs)
            if self.add_lines:
                poly = np.poly1d(np.polyfit(data0, data1, 1))
                y = [poly(d0) for d0 in data0]
                self.ax.plot(data0, y)
        else:
            r = self.calculate(data0, data1)
        return r

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

    @classmethod
    def short_name(cls) -> str:
        raise NotImplementedError

    @classmethod
    def calculate(cls, data0: list, data1: list) -> float:
        """
        calculate and return the correlation value
        """
        raise NotImplementedError
