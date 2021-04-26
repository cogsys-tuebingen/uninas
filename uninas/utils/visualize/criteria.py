import matplotlib.pyplot as plt
import torch
from uninas.training.criteria.common import HuberCriterion, L1Criterion, L2Criterion


if __name__ == '__main__':
    x_center, x_interval = 0, 3
    criteria = [
        HuberCriterion(None, reduction="none", delta=1),
        L1Criterion(None, reduction="none"),
        L2Criterion(None, reduction="none"),
    ]

    x = torch.arange(x_center - x_interval, x_center + x_interval, step=0.05, dtype=torch.float32)
    target = torch.zeros_like(x).add_(x_center)

    for criterion in criteria:
        with torch.no_grad():
            y = criterion(x, target)
            plt.plot(x, y, label=criterion.__class__.__name__)

    plt.ylabel('loss')
    plt.xlabel('difference')
    plt.legend()
    plt.show()
