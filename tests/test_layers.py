import unittest
import torch
import torch.nn as nn
from uninas.methods.strategy_manager import StrategyManager
from uninas.methods.random import RandomChoiceStrategy
from uninas.modules.layers.common import SkipLayer
from uninas.modules.layers.cnn import ZeroLayer, FactorizedReductionLayer, PoolingLayer, ConvLayer, SepConvLayer
from uninas.modules.layers.shufflenet import ShuffleNetV2Layer, ShuffleNetV2XceptionLayer
from uninas.modules.layers.mobilenet import MobileInvertedConvLayer, SharedMixedMobileInvertedConvLayer
from uninas.modules.layers.singlepathnas import SuperConvThresholdLayer, SuperSepConvThresholdLayer, SuperMobileInvertedConvThresholdLayer
from uninas.modules.layers.superkernels import SuperConvLayer, SuperSepConvLayer, SuperMobileInvertedConvLayer
from uninas.modules.layers.scarletnas import LinearTransformerLayer
from uninas.modules.attention.abstract import AttentionLayer
from uninas.modules.modules.shared import AbstractSharedPathsOp
from uninas.utils.shape import Shape
from uninas.builder import Builder


def assert_same_shape(s0: Shape, s1: Shape):
    for i, (s0_, s1_) in enumerate(zip(s0, s1)):
        assert s0_ == s1_, 'Output shape not as expected in dim %d: %s, %s' % (i, s0, s1)


def assert_output_shape(module: nn.Module, x: torch.Tensor, expected_shape: Shape):
    output_ = module(x)
    assert_same_shape(output_.size(), expected_shape)


class TestLayers(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_output_shapes(self):
        """
        expected output shapes of standard layers
        """
        Builder()
        StrategyManager().delete_strategy('default')
        StrategyManager().add_strategy(RandomChoiceStrategy(max_epochs=1))

        bs, c1, c2, hw1, hw2 = 4, 4, 8, 32, 16
        s_in = Shape([c1, hw1, hw1])
        x = torch.empty(size=[bs] + s_in.shape)

        case_s1_c1 = (c1, 1, Shape([c1, hw1, hw1]))
        case_s1_c2 = (c2, 1, Shape([c2, hw1, hw1]))
        case_s2_c1 = (c1, 2, Shape([c1, hw2, hw2]))
        case_s2_c2 = (c2, 2, Shape([c2, hw2, hw2]))

        for cls, cases, kwargs in [
            (SkipLayer,                             [case_s1_c1, case_s1_c2],                         dict()),
            (ZeroLayer,                             [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict()),
            (FactorizedReductionLayer,              [case_s2_c1, case_s2_c2],                         dict()),
            (PoolingLayer,                          [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_size=3)),
            (ConvLayer,                             [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_size=3)),
            (SepConvLayer,                          [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_size=3)),
            (MobileInvertedConvLayer,               [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_size=3, fused=False)),
            (MobileInvertedConvLayer,               [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_size=(3,), fused=False)),
            (MobileInvertedConvLayer,               [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_size=(3, 5, 7), k_size_in=(1, 1), k_size_out=(1, 1), fused=False)),
            (MobileInvertedConvLayer,               [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_size=3, fused=True)),
            (MobileInvertedConvLayer,               [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_size=(3,), fused=True)),
            (MobileInvertedConvLayer,               [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_size=(3, 5, 7), k_size_in=(1, 1), k_size_out=(1, 1), fused=True)),
            (SharedMixedMobileInvertedConvLayer,    [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(name='mmicl1', k_sizes=(3, 5, 7), k_size_in=(1, 1), k_size_out=(1, 1))),
            (SharedMixedMobileInvertedConvLayer,    [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(name='mmicl2', k_sizes=((3, 5), (3, 5, 7)), k_size_in=(1, 1), k_size_out=(1, 1))),
            (ShuffleNetV2Layer,                     [case_s1_c1, case_s1_c2, case_s2_c2],             dict(k_size=3)),
            (ShuffleNetV2XceptionLayer,             [case_s1_c1, case_s1_c2, case_s2_c2],             dict(k_size=3)),
            (LinearTransformerLayer,                [case_s1_c1, case_s1_c2],                         dict()),
            (SuperConvThresholdLayer,               [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_sizes=(3, 5, 7))),
            (SuperSepConvThresholdLayer,            [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_sizes=(3, 5, 7))),
            (SuperMobileInvertedConvThresholdLayer, [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_sizes=(3, 5, 7), expansions=(3, 6), sse_dict=dict(c_muls=(0.0, 0.25, 0.5)))),
            (SuperConvLayer,                        [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_sizes=(3, 5, 7), name='scl')),
            (SuperSepConvLayer,                     [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_sizes=(3, 5, 7), name='sscl')),
            (SuperMobileInvertedConvLayer,          [case_s1_c1, case_s1_c2, case_s2_c1, case_s2_c2], dict(k_sizes=(3, 5, 7), name='smicl', expansions=(3, 6))),
            (AttentionLayer,                        [case_s1_c1],                                     dict(att_dict=dict(att_cls='EfficientChannelAttentionModule'))),
            (AttentionLayer,                        [case_s1_c1],                                     dict(att_dict=dict(att_cls='SqueezeExcitationChannelModule'))),
        ]:
            for c, stride, shape_out in cases:
                m1 = cls(stride=stride, **kwargs)
                s_out = m1.build(s_in, c)
                assert s_out == shape_out, 'Expected output shape does not match, %s, build=%s / expected=%s' %\
                                           (cls.__name__, s_out, shape_out)
                assert_output_shape(m1, x, [bs]+shape_out.shape)
                print('%s(stride=%d, c_in=%d, c_out=%d)' % (cls.__name__, stride, c1, c))

    def test_rebuild(self):
        """
        getting finalized configs from which we can build modules,
        (re)builds module, makes a forward pass
        """
        builder = Builder()
        sm = StrategyManager()
        sm.delete_strategy('default')
        sm.add_strategy(RandomChoiceStrategy(max_epochs=1))
        n, c, h, w = 2, 8, 16, 16
        x = torch.empty(size=[n, c, h, w])
        shape = Shape([c, h, w])
        layers = [
            SharedMixedMobileInvertedConvLayer(name='smmicl', k_sizes=(3, 5, 7), expansions=(3, 6), fused=False),
            SharedMixedMobileInvertedConvLayer(name='smmicl', k_sizes=(3, 5, 7), expansions=(3, 6), fused=True),
            SuperConvThresholdLayer(k_sizes=(3, 5, 7)),
            SuperSepConvThresholdLayer(k_sizes=(3, 5, 7)),
            SuperMobileInvertedConvThresholdLayer(k_sizes=(3, 5, 7), expansions=(3, 6), sse_dict=dict(c_muls=(0.0, 0.25, 0.5))),
            LinearTransformerLayer(),
            SuperConvLayer(k_sizes=(3, 5, 7), name='scl1'),
            SuperSepConvLayer(k_sizes=(3, 5, 7), name='scl2'),
            SuperMobileInvertedConvLayer(k_sizes=(3, 5, 7), name='scl3', expansions=(2, 3, 4, 6)),
        ]
        all_layers = []
        for i, layer in enumerate(layers):
            if isinstance(layer, AbstractSharedPathsOp):
                for m in layer.get_paths_as_modules():
                    all_layers.append(m)
            else:
                all_layers.append(layer)
        for layer in all_layers:
            assert layer.build(shape, c) == shape
        sm.build()
        sm.forward()
        for layer in all_layers:
            print('\n'*2)
            print(layer.__class__.__name__)
            for i in range(3):
                sm.randomize_weights()
                sm.forward()
                for finalize in [False, True]:
                    cfg = layer.config(finalize=finalize)
                    print('\t', i, 'finalize', finalize)
                    print('\t\tconfig dct:', cfg)
                    cfg_layer = builder.from_config(cfg)
                    assert cfg_layer.build(shape, c) == shape
                    cfg_layer.forward(x)
                    print('\t\tmodule str:', cfg_layer.str()[1:])
                    del cfg, cfg_layer


if __name__ == '__main__':
    unittest.main()
