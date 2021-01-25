"""
optimizers from the timm framework
https://github.com/rwightman/pytorch-image-models
"""

from uninas.utils.args import Argument
from uninas.training.optimizers.abstract import WrappedOptimizer
from uninas.register import Register


try:
    from timm.optim.rmsprop_tf import RMSpropTF


    @Register.optimizer()
    class RMSpropTFOptimizer(WrappedOptimizer):
        """
        The TensorFlow-style RMSprop implementation from pytorch-image-models
        """
        optimizer_cls = RMSpropTF

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return [
                Argument('lr', default=0.01, type=float, help='learning rate'),
                Argument('alpha', default=0.99, type=float, help='alpha value'),
                Argument('eps', default=1e-8, type=float, help='epsilon value'),
                Argument('momentum', default=0.0, type=float, help='momentum'),
                Argument('centered', default='False', type=str, help='centered', is_bool=True),
                Argument('decoupled_decay', default='False', type=str, help='decoupled weight decay', is_bool=True),
                Argument('lr_in_momentum', default='True', type=str, help='learning rate scaling is included in the momentum buffer', is_bool=True),
            ] + super().args_to_add(index)

except ImportError as e:
    Register.missing_import(e)
