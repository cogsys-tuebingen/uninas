import torch
import torch.nn as nn
from math import log
from uninas.modules.attention.abstract import AbstractAttentionModule
from uninas.register import Register


@Register.attention_module()
class EfficientChannelAttentionModule(AbstractAttentionModule):
    """
    Efficient Channel Attention
    https://arxiv.org/abs/1910.03151
    https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    """

    def __init__(self, c: int, c_substitute: int = None, use_c_substitute=False,
                 k_size=-1, gamma=2, b=1, excite_act='sigmoid'):
        """
        A cheap and efficient channel attention module,
        implemented according to the paper
        (with an exchangeable excitation function)

        :param c: number of input and output channels
        :param c_substitute: used instead of 'c' for calculating inner channels, if not None and 'use_c_substitute'
                             in MobileNet and ShuffleNet blocks this is the number of block input channels
                             (usually fewer than the input channels of the SE module within the block)
        :param use_c_substitute: try using 'c_substitute'
        :param k_size: 1D convolution kernel size, adaptive if <= 0
        :param gamma: parameter to choose k_size adaptively
        :param b: parameter to choose k_size adaptively
        :param excite_act: activation function after exciting
        """
        super().__init__(c, c_substitute, use_c_substitute)
        k_size = self.map_fun(self.c, k_size, gamma=gamma, b=b)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.excite_act = Register.act_funs.get(excite_act)(inplace=True)

    @classmethod
    def map_fun(cls, c: int, k_size=-1, gamma=2, b=2) -> int:
        if k_size < 0:
            k_size = int(abs((log(c, 2) + b) / gamma))
        return k_size if k_size % 2 == 1 else k_size + 1  # make k_size odd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gap(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.excite_act(y)
        return x * y.expand_as(x)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x_ = list(range(1, 10000+1))
    y_ = [EfficientChannelAttentionModule.map_fun(xi, -1, gamma=2, b=1) for xi in x_]
    plt.plot(x_, y_)
    plt.xscale('log')
    plt.show()
