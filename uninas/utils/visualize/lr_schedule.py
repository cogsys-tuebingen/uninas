import matplotlib.pyplot as plt
import torch
from uninas.utils.args import ArgumentParser
from uninas.training.optimizers.common import SGDOptimizer as Optimizer
from uninas.training.schedulers.common import CosineScheduler as Scheduler


if __name__ == '__main__':
    max_epochs = 50
    fake_max_epochs = -1
    cooldown_epochs = 5
    warmup_epochs = 5
    warmup_lr = 0.0
    init_lr = 0.25
    min_lr = 0.05

    a = torch.nn.Parameter(data=torch.zeros(size=(1,)), requires_grad=True)

    p = ArgumentParser()
    Optimizer.add_arguments(p)
    Scheduler.add_arguments(p)

    args = p.parse_args()
    args.__setattr__('%s.lr' % Optimizer.__name__, init_lr)
    args.__setattr__('%s.cooldown_epochs' % Scheduler.__name__, cooldown_epochs)
    args.__setattr__('%s.warmup_epochs' % Scheduler.__name__, warmup_epochs)
    args.__setattr__('%s.fake_num_epochs' % Scheduler.__name__, fake_max_epochs)
    args.__setattr__('%s.warmup_lr' % Scheduler.__name__, warmup_lr)
    args.__setattr__('%s.eta_min' % Scheduler.__name__, min_lr)

    opt = Optimizer.from_args(args, index=None, named_params=[('a', a)])
    scheduler = Scheduler.from_args(args, opt, max_epochs)
    print(args)

    print(opt)
    print(scheduler)

    x = list(range(max_epochs))
    y = []
    for xs in x:
        y.append(opt.param_groups[0]['lr'])
        scheduler.step()

    print(y)
    plt.plot(x, y)
    plt.ylim((0, init_lr))
    plt.ylabel('lr')
    plt.xlabel('epoch')
    plt.show()
