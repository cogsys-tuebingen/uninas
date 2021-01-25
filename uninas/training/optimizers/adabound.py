from uninas.register import Register
from uninas.utils.args import Argument
from uninas.training.optimizers.abstract import WrappedOptimizer
try:
    from adabound import AdaBound


    @Register.optimizer()
    class AdaBoundOptimizer(WrappedOptimizer):
        """
        Adaptive bounds on the learning rate, transitioning from Adam to SGD

        https://arxiv.org/abs/1902.09843
        https://openreview.net/forum?id=Bkg3g2R9FX
        https://github.com/Luolc/AdaBound
        """

        @classmethod
        def optimizer_cls(cls, params=None, lr=0.001, beta1=0.9, beta2=0.999, final_lr=0.1, eps=1e-8, weight_decay=1e-5, gamma=1e-3, amsbound=False):
            return AdaBound(params=params, lr=lr, betas=(beta1, beta2), final_lr=final_lr, eps=eps, weight_decay=weight_decay, gamma=gamma, amsbound=amsbound)

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return [
                Argument('lr', default=0.001, type=float, help='adam learning rate'),
                Argument('beta1', default=0.9, type=float, help='adam beta1 value'),
                Argument('beta2', default=0.999, type=float, help='adam beta2 value'),
                Argument('final_lr', default=0.1, type=float, help='final sgd learning rate'),
                Argument('gamma', default=0.001, type=float, help='speed of the bound functions'),
                Argument('eps', default=1e-8, type=float, help='epsilon value for numerical stability'),
                Argument('amsbound', default='False', type=str, help='use amsbound variant', is_bool=True),
            ] + super().args_to_add(index)

except ImportError as e:
    Register.missing_import(e)
