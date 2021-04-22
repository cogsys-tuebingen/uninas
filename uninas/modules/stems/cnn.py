from uninas.modules.stems.abstract import SingleLayerStem
from uninas.modules.layers.cnn import ConvLayer
from uninas.utils.args import Argument
from uninas.register import Register


@Register.network_stem()
class ConvStem(SingleLayerStem):
    """ a simple conv layer as stem """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('stride', default=1, type=int, help='stride'),
            Argument('k_size', default=3, type=int, help='kernel size'),
            Argument('act_fun', default='identity', type=str, help='activation function', choices=Register.act_funs.names()),
            Argument('order', default='w_bn_act', type=str, help='order of act fun, batch norm, weights'),
            Argument('use_bn', default='True', type=str, help='use batchnorm', is_bool=True),
            Argument('bn_affine', default='True', type=str, help='affine batchnorm', is_bool=True),
        ]

    @classmethod
    def stem_from_kwargs(cls, **kwargs) -> SingleLayerStem:
        stem = ConvLayer(k_size=kwargs.get('k_size'), dilation=1, stride=kwargs.get('stride'),
                         act_fun=kwargs.get('act_fun'), dropout_rate=0.0,
                         use_bn=kwargs.get('use_bn'), bn_affine=kwargs.get('bn_affine'), order=kwargs.get('order'))
        return cls(stem, **kwargs)
