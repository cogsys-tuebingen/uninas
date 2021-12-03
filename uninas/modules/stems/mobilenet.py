import torch
from uninas.modules.stems.abstract import AbstractStem
from uninas.modules.modules.abstract import AbstractModule
from uninas.modules.layers.cnn import ConvLayer
from uninas.modules.layers.mobilenet import MobileInvertedConvLayer
from uninas.utils.args import Argument
from uninas.utils.shape import Shape, ShapeList
from uninas.register import Register


@Register.network_stem()
class MobileNetV2Stem(AbstractStem):
    """ a simple conv layer as stem, followed by an MobileInvertedConvLayer with expansion 1.0 """
    _num_outputs = 1

    def __init__(self, stem0: AbstractModule, stem1: AbstractModule, **stored_kwargs):
        super().__init__(**stored_kwargs)
        self._add_to_submodules(stem0=stem0, stem1=stem1)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('stride', default=2, type=int, help='stride of the conv layer'),
            Argument('k_size', default=3, type=int, help='kernel size of the conv layer'),
            Argument('act_fun', default='relu6', type=str, help='act fun of the conv layer', choices=Register.act_funs.names()),
            Argument('features1', default=16, type=int, help='num output features the mobilenet layer'),
            Argument('stride1', default=1, type=int, help='stride of the mobilenet layer'),
            Argument('k_size1', default=3, type=int, help='kernel size of the mobilenet layer'),
            Argument('act_fun1', default='relu6', type=str, help='act fun of the mobilenet layer', choices=Register.act_funs.names()),
            Argument('se_cmul1', default=0.0, type=float, help='use Squeeze+Excitation with act_fun1 and given width'),
        ]

    @classmethod
    def stem_from_kwargs(cls, **kwargs) -> AbstractStem:
        stem0 = ConvLayer(k_size=kwargs.get('k_size'), dilation=1, stride=kwargs.get('stride'),
                          act_fun=kwargs.get('act_fun'), act_inplace=True,
                          dropout_rate=0.0, use_bn=True, bn_affine=True, order='w_bn_act')
        act_fun1 = kwargs.get('act_fun1')
        se_cmul1 = kwargs.get('se_cmul1')
        att_dict = dict(att_cls='SqueezeExcitationChannelModule', squeeze_act=act_fun1, c_mul=se_cmul1)\
            if se_cmul1 > 0 else None
        stem1 = MobileInvertedConvLayer(k_size=kwargs.get('k_size1'), dilation=1, stride=kwargs.get('stride1'),
                                        act_fun=act_fun1, bn_affine=True, expansion=1.0, att_dict=att_dict, fused=False)
        return cls(stem0, stem1, **kwargs)

    def _build(self, s_in: Shape) -> ShapeList:
        """ build the stem for the data set, return list of output feature sizes """
        s0 = self.stem0.build(s_in, self.features)
        self.cached['shape_inner'] = s0
        s1 = self.stem1.build(s0, self.features1)
        return ShapeList([s1])

    def forward(self, x: torch.Tensor) -> [torch.Tensor]:
        return [self.stem1(self.stem0(x))]
