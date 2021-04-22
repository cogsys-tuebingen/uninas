"""
easily generate specific network configs from the search net task config and a sequence of primitive indices
or via a function that builds a network the regular way
"""

import argparse
import torch.nn as nn
from uninas.builder import Builder
from uninas.main import Main
from uninas.utils.paths import get_net_config_dir
from uninas.utils.args import Namespace
from uninas.utils.generate.networks.manually.resnet import get_resnet18, get_resnet34, get_resnet50, get_resnet101
from uninas.utils.generate.networks.manually.mobilenet import get_mobilenet_v3_large100, get_mobilenet_v3_small100
from uninas.utils.generate.networks.manually.shufflenet import get_shufflenet_v2plus_medium
from uninas.utils.generate.networks.manually.mixnet import get_mixnet_s, get_mixnet_m


class NetWrapper:

    def generate(self, save_dir: str, name: str, verbose=True) -> (nn.Module, dict, Namespace):
        raise NotImplementedError

    @staticmethod
    def save(net: nn.Module, save_dir: str, name: str, verbose=True) -> dict:
        # saving config now will only use the currently active connections, since we have a search network
        cfg = net.config(finalize=True)
        if (save_dir is not None) and (name is not None):
            path = Builder.save_config(cfg, save_dir, name)
            if verbose:
                print('Saved config: %s' % path)
        return cfg


class Genotype(NetWrapper):
    def __init__(self, search_net: str, gene: [int], source='original'):
        self.search_net = search_net
        self.gene = gene
        self.source = source

    def generate(self, save_dir: str, name: str, verbose=True) -> (nn.Module, dict, Namespace):
        run_configs = '{path_conf_tasks}/s1_random.run_config, {path_conf_net_search}/%s.run_config' % self.search_net
        task = Main.new_task(run_configs, args_changes={
            '{cls_data}.fake': True,
            '{cls_task}.save_del_old': False,
            '{cls_task}.save_dir': '{path_tmp}/generate/',
        })
        net = task.get_method().get_network()
        args = task.args

        # fix arc
        net.forward_strategy(fixed_arc=self.gene)

        cfg = self.save(net, save_dir, name, verbose)
        return net, cfg, args


class BuildFun(NetWrapper):
    def __init__(self, function, source='original'):
        self.function = function
        self.source = source

    def generate(self, save_dir: str, name: str, verbose=True) -> (nn.Module, dict, Namespace):
        net = self.function()
        cfg = self.save(net, save_dir, name, verbose)
        return net, cfg, None


# ------------------------------------------------------------------------------------------------------------------
# manually created or discovered in other frameworks

ResNet18 = BuildFun(get_resnet18, source='originals/resnet')
ResNet34 = BuildFun(get_resnet34, source='originals/resnet')
ResNet50 = BuildFun(get_resnet50, source='originals/resnet')
ResNet101 = BuildFun(get_resnet101, source='originals/resnet')

MobileNetV2 = Genotype(
    search_net='mobilenet_v2',
    gene=[3]*16,
    source='originals/mobilenet'
)

MobileNetV3Large100 = BuildFun(get_mobilenet_v3_large100, source='originals/mobilenet')
MobileNetV3Small100 = BuildFun(get_mobilenet_v3_small100, source='originals/mobilenet')

EfficientNetB0 = Genotype(
    search_net='efficientnet',
    gene=[3, 3,   4, 4,   3, 3, 3,   4, 4, 4,   4, 4, 4, 4,   3],
    source='originals/mobilenet'
)

MixNetS = BuildFun(get_mixnet_s, source='originals/mobilenet')
MixNetM = BuildFun(get_mixnet_m, source='originals/mobilenet')

ShuffleNetV2PlusMedium = BuildFun(get_shufflenet_v2plus_medium, source='originals/shufflenet')
SPOSNet = Genotype(
    # https://github.com/megvii-model/SinglePathOneShot
    search_net='shufflenet_v2',
    gene=[2, 1, 0, 1,   2, 0, 2, 0,   2, 0, 2, 3, 0, 0, 0, 0,   3, 2, 3, 3],
    source='originals/shufflenet',
)

ProxylessRMobile = Genotype(
    # https://github.com/mit-han-lab/proxylessnas/blob/master/proxyless_nas/model_zoo.py
    search_net='proxylessnas',
    gene=[1, 0, 6, 6,   2, 0, 1, 1,   5, 1, 1, 1,   4, 1, 1, 1,   5, 5, 2, 2,   5],
    source='originals/mobilenet',
)

FairNasA = Genotype(
    # https://github.com/xiaomi-automl/FairNAS/blob/master/models/FairNAS_A.py
    search_net='fairnas',
    gene=[2, 0,   2, 3, 5, 0,   0, 5, 5, 1,   3, 1, 1, 3,   3, 5, 3, 5,   4],
    source='originals/mobilenet',
)
FairNasB = Genotype(
    # https://github.com/xiaomi-automl/FairNAS/blob/master/models/FairNAS_B.py
    search_net='fairnas',
    gene=[1, 0,   1, 0, 3, 1,   2, 0, 3, 1,   0, 3, 2, 0,   5, 4, 5, 3,   4],
    source='originals/mobilenet',
)
FairNasC = Genotype(
    # https://github.com/xiaomi-automl/FairNAS/blob/master/models/FairNAS_C.py
    search_net='fairnas',
    gene=[1, 0,   2, 0, 0, 0,   0, 0, 0, 3,   0, 0, 0, 0,   5, 5, 3, 3,   4],
    source='originals/mobilenet',
)

ScarletNasA = Genotype(
    # https://github.com/xiaomi-automl/SCARLET-NAS/blob/master/models/Scarlet_A.py
    search_net='scarletnas',
    gene=[2, 4,   1, 8, 9, 7,   9, 0, 8, 2,   1, 8, 0, 8,   9, 10, 6, 9,   11],
    source='originals/mobilenet',
)
ScarletNasB = Genotype(
    # https://github.com/xiaomi-automl/SCARLET-NAS/blob/master/models/Scarlet_B.py
    search_net='scarletnas',
    gene=[6, 7,   6, 11, 0, 1,   11, 6, 12, 1,   6, 6, 11, 6,   10, 10, 12, 11,   10],
    source='originals/mobilenet',
)
ScarletNasC = Genotype(
    # https://github.com/xiaomi-automl/SCARLET-NAS/blob/master/models/Scarlet_C.py
    search_net='scarletnas',
    gene=[7, 6,   7, 12, 12, 0,   11, 6, 6, 1,   8, 2, 6, 8,   6, 12, 9, 11,   10],
    source='originals/mobilenet',
)


# ------------------------------------------------------------------------------------------------------------------
# discovered in uninas

Net_dna_298325_1_4 = Genotype(
    # dna1: 298325, dna2: 300507+302358, pareto nr: 4, ~271m macs
    search_net='fairnas',
    gene=[1, 1,   0, 0, 0, 0,   1, 2, 1, 1,   2, 1, 1, 0,   4, 1, 0, 1,   0],
    source='discovered/fairnas/dna',
)

Net_dna_298325_1_11 = Genotype(
    # dna1: 298325, dna2: 300507+302358, pareto nr: 11, ~289m macs
    search_net='fairnas',
    gene=[2, 3,   1, 0, 1, 0,   1, 1, 0, 0,   1, 2, 1, 2,   1, 2, 2, 2,   2],
    source='discovered/fairnas/dna',
)

Net_dna_298325_1_13 = Genotype(
    # dna1: 298325, dna2: 300507+302358, pareto nr: 13, ~301m macs
    search_net='fairnas',
    gene=[1, 3,   1, 0, 1, 1,   4, 0, 3, 0,   0, 0, 0, 0,   4, 1, 1, 2,   1],
    source='discovered/fairnas/dna',
)

Net_dna_298325_1_17 = Genotype(
    # dna1: 298325, dna2: 300507+302358, pareto nr: 17, ~322m macs
    search_net='fairnas',
    gene=[2, 3,   1, 0, 1, 1,   1, 2, 1, 1,   4, 2, 1, 2,   4, 1, 2, 4,   2],
    source='discovered/fairnas/dna',
)

Net_dna_298325_1_27 = Genotype(
    # dna1: 298325, dna2: 300507+302358, pareto nr: 27, ~346m macs
    search_net='fairnas',
    gene=[1, 4,   1, 1, 1, 1,   4, 1, 4, 1,   1, 2, 1, 2,   4, 1, 4, 5,   2],
    source='discovered/fairnas/dna',
)


# ------------------------------------------------------------------------------------------------------------------

def generate_from_name(genotype_name: str, verbose=True):
    genotype = globals()[genotype_name]
    return genotype.generate(get_net_config_dir(genotype.source), genotype_name, verbose)


def main():
    Builder()
    parser = argparse.ArgumentParser(description='generate a network config from simple genotype description')
    parser.add_argument('--genotypes', type=str, default=None, help='which config to generate, all available if None')
    args = parser.parse_args()

    if args.genotypes is not None:
        all_genotype_names = [args.genotypes]
    else:
        all_genotype_names = []
        for key, value in list(globals().items()):
            if isinstance(value, NetWrapper):
                all_genotype_names.append(key)

    for genotype_name in all_genotype_names:
        print('Name:\t\t%s' % genotype_name)
        generate_from_name(genotype_name)


if __name__ == '__main__':
    main()
