from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.optim.rmsprop import RMSprop
from uninas.utils.args import Argument
from uninas.training.optimizers.abstract import WrappedOptimizer
from uninas.register import Register


@Register.optimizer()
class SGDOptimizer(WrappedOptimizer):
    optimizer_cls = SGD

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return [
            Argument('lr', default=0.01, type=float, help='learning rate'),
            Argument('momentum', default=0.9, type=float, help='momentum'),
            Argument('nesterov', default='False', type=str, help='use nesterov', is_bool=True),
        ] + super().args_to_add(index)


@Register.optimizer()
class RMSPropOptimizer(WrappedOptimizer):
    optimizer_cls = RMSprop

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return [
            Argument('lr', default=0.01, type=float, help='learning rate'),
            Argument('alpha', default=0.99, type=float, help='alpha value'),
            Argument('eps', default=1e-8, type=float, help='epsilon value'),
            Argument('momentum', default=0.0, type=float, help='momentum'),
            Argument('centered', default='False', type=str, help='centered', is_bool=True),
        ] + super().args_to_add(index)


@Register.optimizer()
class AdamOptimizer(WrappedOptimizer):

    @classmethod
    def optimizer_cls(cls, params=None, lr=0.01, beta1=0.0, beta2=0.0, eps=1e-8, weight_decay=1e-5, amsgrad=False):
        return Adam(params=params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return [
            Argument('lr', default=0.01, type=float, help='learning rate'),
            Argument('eps', default=1e-8, type=float, help='epsilon value'),
            Argument('beta1', default=0.9, type=float, help='beta1 value'),
            Argument('beta2', default=0.999, type=float, help='beta2 value'),
            Argument('amsgrad', default='False', type=str, help='use amsgrad', is_bool=True),
        ] + super().args_to_add(index)
