import unittest
from uninas.utils.generate.networks.darts_configs import generate_from_name as generate_darts_from_name
from uninas.utils.generate.networks.super_configs import generate_from_name as generate_super_from_name
from uninas.utils.torch.misc import count_parameters
from uninas.utils.paths import replace_standard_paths
from uninas.builder import Builder
from uninas.main import Main


retrain_darts_cifar_config = '{path_conf_tasks}/d2_cifar.run_config'
retrain_darts_imagenet_config = '{path_conf_tasks}/d2_imagenet.run_config'
retrain_super_config = '{path_conf_tasks}/s3.run_config'


def assert_stats_match(name, task_cfg, cfg: dict, num_params=None, num_macs=None):
    cfg_dir = replace_standard_paths('{path_tmp}/tests/cfgs/')
    cfg_path = Builder.save_config(cfg, cfg_dir, name)
    exp = Main.new_task(task_cfg, args_changes={
        '{cls_data}.fake': True,
        '{cls_data}.batch_size_train': 2,
        '{cls_data}.batch_size_test': -1,
        '{cls_task}.is_test_run': True,
        '{cls_task}.save_dir': '{path_tmp}/tests/workdir/',
        "{cls_network}.config_path": cfg_path,
        'cls_network_heads': 'ClassificationHead',  # necessary for the DARTS search space to disable the aux heads
    }, raise_unparsed=False)
    net = exp.get_method().get_network()
    macs = exp.get_method().profile_macs()
    net.eval()
    # print(net)
    cp = count_parameters(net)
    if num_params is not None:
        assert cp == num_params, 'Got unexpected num params for %s: %d, expected %d, diff: %d'\
                                 % (name, cp, num_params, abs(cp - num_params))
    if num_macs is not None:
        assert macs == num_macs, 'Got unexpected num macs for %s: %d, expected %d, diff: %d'\
                                 % (name, macs, num_macs, abs(macs - num_macs))


def assert_darts_cifar10_stats_match(name, num_params=None, num_macs=None):
    """ generates darts-like configs and tests them """
    _, cfg, _ = generate_darts_from_name(name, verbose=False)
    assert_stats_match(name, retrain_darts_cifar_config, cfg, num_params, num_macs)


def assert_darts_imagenet_stats_match(name, num_params=None, num_macs=None):
    """ generates darts-like configs and tests them """
    _, cfg, _ = generate_darts_from_name(name, verbose=False)
    assert_stats_match(name, retrain_darts_imagenet_config, cfg, num_params, num_macs)


def assert_super_stats_match(name, num_params=None, num_macs=None):
    """ tests super-net configs """
    _, cfg, _ = generate_super_from_name(name, verbose=False)
    assert_stats_match(name, retrain_super_config, cfg, num_params, num_macs)


class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_model_params(self):
        """
        make sure that the models from known network_configs / genotypes have correct number of params
        Imagenet1k uses input size (3, 224, 224)

        differences to originals:
            (1) they use x.mean(3).mean(2) while we use nn.AdaptiveAvgPool2d(1)
            (2) We use the new torch 1.6 swish/hswish/htanh/hsigmoid activation functions
            (3) TODO marginal macs difference for SPOS after changing search-network/primitives code (maybe act fun?)
        the numbers matched exactly when these were accounted for
        """
        Builder()

        # measured via torch.hub
        assert_super_stats_match('ResNet18',                num_params=11689512, num_macs=1814073856)
        assert_super_stats_match('ResNet34',                num_params=21797672, num_macs=3663761920)
        assert_super_stats_match('ResNet50',                num_params=25557032, num_macs=4089186304)
        assert_super_stats_match('ResNet101',               num_params=44549160, num_macs=7801407488)
        assert_super_stats_match('MobileNetV2',             num_params=3504872, num_macs=300775552)

        # measured via https://github.com/megvii-model/SinglePathOneShot
        assert_super_stats_match('SPOSNet',                 num_params=3558464, num_macs=322919776-16)  # (3)

        # measured via https://github.com/megvii-model/ShuffleNet-Series
        assert_super_stats_match('ShuffleNetV2PlusMedium',  num_params=5679840, num_macs=224038432-1531648-16)  # (2), (3)

        # measured via https://github.com/rwightman/pytorch-image-models
        # requires replacing the swish function, otherwise torchprofile tracing fails
        assert_super_stats_match('EfficientNetB0',          num_params=5288548, num_macs=394289436)
        assert_super_stats_match('MobileNetV3Large100',     num_params=5483032, num_macs=218703448-1511264)  # (2)
        assert_super_stats_match('MobileNetV3Small100',     num_params=2542856, num_macs=57597784-799136)    # (2)
        assert_super_stats_match('MixNetS',                 num_params=4134606, num_macs=None)
        assert_super_stats_match('MixNetM',                 num_params=5014382, num_macs=None)

        # measured via https://github.com/mit-han-lab/proxylessnas
        assert_super_stats_match('ProxylessRMobile',        num_params=4080512, num_macs=320428864)

        # measured via https://github.com/xiaomi-automl/FairNAS
        assert_super_stats_match('FairNasA',                num_params=4651352, num_macs=388133088-8960)  # (1)
        assert_super_stats_match('FairNasB',                num_params=4506272, num_macs=345307872-8960)  # (1)
        assert_super_stats_match('FairNasC',                num_params=4397864, num_macs=321035232-8960)  # (1)

        # measured via https://github.com/xiaomi-automl/SCARLET-NAS
        assert_super_stats_match('ScarletNasA',             num_params=6707720, num_macs=370705152-8960-4572288)  # (1, 2)
        assert_super_stats_match('ScarletNasB',             num_params=6531556, num_macs=332082096-8960-3932544)  # (1, 2)
        assert_super_stats_match('ScarletNasC',             num_params=6073684, num_macs=284388720-8960-3278688)  # (1, 2)

        # measured via https://github.com/cogsys-tuebingen/prdarts
        assert_darts_cifar10_stats_match('DARTS_V1',        num_params=3169414, num_macs=501015744)
        assert_darts_cifar10_stats_match('PDARTS',          num_params=3433798, num_macs=532202688)
        assert_darts_cifar10_stats_match('PR_DARTS_DL1',    num_params=3174166, num_macs=491200704)
        assert_darts_cifar10_stats_match('PR_DARTS_DL2',    num_params=4017646, num_macs=650370240)
        assert_darts_imagenet_stats_match('DARTS_V1',       num_params=4510432, num_macs=505893696)
        assert_darts_imagenet_stats_match('PDARTS',         num_params=4944352, num_macs=542848320)
        assert_darts_imagenet_stats_match('PR_DARTS_DL1',   num_params=4685152, num_macs=509384064)
        assert_darts_imagenet_stats_match('PR_DARTS_DL2',   num_params=5529856, num_macs=631603392)

        # measured via https://github.com/tanglang96/MDENAS
        assert_darts_cifar10_stats_match('MdeNAS',          num_params=3786742, num_macs=599110848)
        assert_darts_imagenet_stats_match('MdeNAS',         num_params=5329024, num_macs=595514304)

        # bench201 would be nice


if __name__ == '__main__':
    unittest.main()
