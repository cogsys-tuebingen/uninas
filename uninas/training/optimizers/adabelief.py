from uninas.register import Register
from uninas.utils.args import Argument
from uninas.training.optimizers.abstract import WrappedOptimizer
try:
    from adabelief_pytorch import AdaBelief


    @Register.optimizer()
    class AdaBeliefOptimizer(WrappedOptimizer):
        """
        https://juntang-zhuang.github.io/adabelief/
        https://arxiv.org/abs/2010.07468
        https://github.com/juntang-zhuang/Adabelief-Optimizer
        """

        @classmethod
        def optimizer_cls(cls, params=None, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-5,
                          amsgrad=False, weight_decouple=False, fixed_decay=False, rectify=False):
            return AdaBelief(params=params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay,
                             amsgrad=amsgrad, weight_decouple=weight_decouple, fixed_decay=fixed_decay, rectify=rectify)

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return [
                Argument('lr', default=1e-3, type=float, help='adam learning rate'),
                Argument('beta1', default=0.9, type=float, help='adam beta1 value'),
                Argument('beta2', default=0.999, type=float, help='adam beta2 value'),
                Argument('eps', default=1e-8, type=float, help='epsilon value for numerical stability'),
                Argument('amsgrad', default="False", type=str, is_bool=True,
                         help='whether to use the AMSGrad variant of this algorithm from the paper'
                              '`On the Convergence of Adam and Beyond`_'),
                Argument('weight_decouple', default="False", type=str, is_bool=True,
                         help='If set as True, then the optimizer uses decoupled weight decay as in AdamW'),
                Argument('fixed_decay', default="False", type=str, is_bool=True,
                         help='This is used when weight_decouple is set as True.\n'
                              'When fixed_decay == True, the weight decay is performed as $W_{new} = W_{old} - W_{old} * decay$.\n'
                              'When fixed_decay == False, the weight decay is performed as $W_{new} = W_{old} - W_{old} * decay * lr$.\n'
                              'Note that in this case, the weight decay ratio decreases with learning rate (lr).'),
                Argument('rectify', default="False", type=str, is_bool=True,
                         help='If set as True, then perform the rectified update similar to RAdam'),
            ] + super().args_to_add(index)

except ImportError as e:
    Register.missing_import(e)
