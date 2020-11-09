"""
https://arxiv.org/abs/2001.00326
https://github.com/D-X-Y/NAS-Bench-201

the most relevant parts of the original API, requires significantly less RAM (few MB << 25+ GB)
averages training results of the last few epochs, also averages flops/latency, also enables querying via tuples
"""

import os
from collections import defaultdict
import numpy as np
import torch
from nas_201_api import ArchResults as ArchResults201, NASBench201API as API201
from uninas.optimization.hpo_self.values import DiscreteValues, ValueSpace
from uninas.model.primitives.bench201 import Bench201Primitives
from uninas.utils.loggers.exp import LightningLoggerBase
from uninas.utils.paths import replace_standard_paths


class MiniResult:
    """
    a result subset that contains only the latest loss/acc1,
    all values are averaged over multiple runs/seeds by the topology
    """

    def __init__(self, arch_index: int, arch_str: str, arch_tuple: tuple,
                 params: dict, flops: dict, latency: dict, loss: dict, acc1: dict):
        self.arch_index = arch_index
        self.arch_str = arch_str
        self.arch_tuple = arch_tuple
        self.params = params
        self.flops = flops
        self.latency = latency
        self.loss = loss
        self.acc1 = acc1

    def state_dict(self) -> dict:
        dct = dict()
        dct.update(self.named_values())
        dct.update(self.named_dicts())
        return dct

    def named_values(self) -> dict:
        return dict(
            arch_index=self.arch_index,
            arch_str=self.arch_str,
            arch_tuple=self.arch_tuple,
        )

    def named_dicts(self) -> {str: dict}:
        return dict(
            params=self.params,
            flops=self.flops,
            latency=self.latency,
            loss=self.loss,
            acc1=self.acc1,
        )

    @classmethod
    def make_from_result201(cls, result: ArchResults201, arch_tuple: tuple, epoch_keys=(197, 198, 199)):
        params = defaultdict(list)
        flops = defaultdict(list)
        latency = defaultdict(list)
        losses = defaultdict(list)
        acc1 = defaultdict(list)
        for (data_set, seed), single_result in result.all_results.items():
            params[data_set].append(single_result.params)
            flops[data_set].append(single_result.flop)
            latency[data_set].extend(single_result.latency)
            for k in epoch_keys:
                losses[data_set].append(single_result.train_losses[k])
                acc1[data_set].append(single_result.train_acc1es[k])

        return MiniResult(
            arch_index=result.arch_index,
            arch_str=result.arch_str,
            arch_tuple=arch_tuple,
            params={k: sum(v) / len(v) for k, v in params.items()},
            flops={k: sum(v) / len(v) for k, v in flops.items()},
            latency={k: sum(v) / len(v) for k, v in latency.items()},
            loss={k: sum(v) / len(v) for k, v in losses.items()},
            acc1={k: sum(v) / len(v) for k, v in acc1.items()},
        )

    def get_info_str(self, data_set: str) -> str:
        return "{cls}(index={idx}, acc1={acc1}, loss={loss}, params={par}, flops={flops}, latency={lat})".format(**{
            'cls': self.__class__.__name__,
            'idx': str(self.arch_index),
            'acc1': str(self.acc1.get(data_set)),
            'loss': str(self.loss.get(data_set)),
            'par': str(self.loss.get(data_set)),
            'flops': str(self.flops.get(data_set)),
            'lat': str(self.latency.get(data_set)),
        })

    def get(self, kind: str, data_set: str) -> float:
        fun = {
            'params': self.get_params,
            'flops': self.get_flops,
            'latency': self.get_latency,
            'loss': self.get_loss,
            'acc1': self.get_acc1,
        }[kind]
        return fun(data_set)

    def get_params(self, data_set: str) -> float:
        return self.params.get(data_set)

    def get_flops(self, data_set: str) -> float:
        return self.flops.get(data_set)

    def get_latency(self, data_set: str) -> float:
        return self.latency.get(data_set)

    def get_loss(self, data_set: str) -> float:
        return self.loss.get(data_set)

    def get_acc1(self, data_set: str) -> float:
        return self.acc1.get(data_set)

    def print(self, print_fun=print, prefix=''):
        for n, v in self.state_dict().items():
            print_fun('{}{:<13}{:}'.format(prefix, n, v))


class MiniNASBenchApi:
    def __init__(self, bench_type: str, results: dict, arch_to_idx: dict, tuple_to_str: dict, tuple_to_idx: dict):
        self.bench_type = bench_type
        self.results = results
        self.arch_to_idx = arch_to_idx
        self.tuple_to_str = tuple_to_str
        self.tuple_to_idx = tuple_to_idx

    def save(self, save_path: str):
        if isinstance(save_path, str):
            save_path = os.path.abspath(replace_standard_paths(save_path))
            name = os.path.basename(save_path)
            path = os.path.dirname(save_path)
            os.makedirs(path, exist_ok=True)
            s = '%s/%s' % (path, name)
            data = dict(
                bench_type=self.bench_type,
                results={k: r.state_dict() for k, r in self.results.items()},
                arch_to_idx=self.arch_to_idx,
                tuple_to_str=self.tuple_to_str,
                tuple_to_idx=self.tuple_to_idx)
            torch.save(data, s)
            print('Saved to %s' % s)

    @classmethod
    def load(cls, path: str):
        dct = torch.load(replace_standard_paths(path))
        dct['results'] = {k: MiniResult(**v) for k, v in dct['results'].items()}
        return MiniNASBenchApi(**dct)

    @classmethod
    def make_from_full_api201(cls, api: API201):
        str_to_idx = {k: i for i, k in enumerate(Bench201Primitives.get_order())}
        results = {}
        arch_to_idx = {}
        tuple_to_str = {}
        tuple_to_idx = {}

        for i, arch_result in api.arch2infos_full.items():
            # name to tuple, tuple to name and index
            ops_str = arch_result.arch_str[1:-1].replace('|+', '').split('|')
            ops_str = [s.split('~')[0] for s in ops_str]
            ops_tuple = tuple([str_to_idx.get(s) for s in ops_str])
            tuple_to_str[ops_tuple] = ops_str
            tuple_to_idx[ops_tuple] = i
            # result
            arch_to_idx[arch_result.arch_str] = i
            results[i] = MiniResult.make_from_result201(arch_result, ops_tuple)

        return MiniNASBenchApi(bench_type="201", results=results, arch_to_idx=arch_to_idx,
                               tuple_to_str=tuple_to_str, tuple_to_idx=tuple_to_idx)

    def get_space(self) -> ValueSpace:
        return {
            '201': ValueSpace(*[DiscreteValues.interval(0, 5) for _ in range(6)])
        }[self.bench_type]

    def get_space_lower_upper(self) -> (np.array, np.array):
        # for pymoo
        return {
            '201': (np.array([0, 0, 0, 0, 0, 0]), np.array([4, 4, 4, 4, 4, 4])),
        }[self.bench_type]

    def get_all(self) -> [MiniResult]:
        return [r for r in self.results.values()]

    def get_all_sorted(self, sorted_by='acc1', data_set='cifar10', reverse=True) -> list:
        return sorted(self.get_all(), key=lambda s: s.get(sorted_by, data_set), reverse=reverse)

    def get_by_index(self, index: int) -> MiniResult:
        return self.results.get(index)

    def get_by_arch_str(self, arch_str: str) -> MiniResult:
        return self.results.get(self.arch_to_idx.get(arch_str))

    def get_by_arch_tuple(self, arch_tuple: tuple) -> MiniResult:
        arch_tuple = tuple([(v[0] if len(v) == 1 else tuple(v)) if isinstance(v, (tuple, list)) else v
                            for v in arch_tuple])  # flatten if possible
        return self.results.get(self.tuple_to_idx.get(arch_tuple))

    def print_info(self, print_fun=print):
        print_fun('{cls}(type="{type}", num_results={nr})'.format(**{
            'cls': self.__class__.__name__,
            'type': self.bench_type,
            'nr': len(self.results),
        }))

    def print_some(self, n=5, print_fun=print):
        for i, r in self.results.items():
            print_fun('-'*180)
            r.print(print_fun=print_fun)
            if i+1 >= n:
                break

    def log_result(self, result: MiniResult, logger: LightningLoggerBase):
        fmt = "bench_{bt}/{ds}/{key}"
        log_dct = dict()
        for key, dct in result.named_dicts().items():
            for ds, v in dct.items():
                log_dct[fmt.format(bt=self.bench_type, ds=ds, key=key)] = v
        logger.log_metrics(log_dct, step=200)


def make_mini201(path_full: str, path_save: str):
    api = API201(path_full)
    mini = MiniNASBenchApi.make_from_full_api201(api)
    mini.save(path_save)


def explore(mini: MiniNASBenchApi):
    print(mini.get_by_arch_tuple((4, 3, 2, 1, 0, 2)).get_info_str('cifar10'))
    mini.get_by_arch_tuple((1, 2, 1, 2, 3, 4)).print()
    mini.get_by_index(1554).print()
    mini.get_by_arch_str('|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|').print()
    # mini.save('{path_data}/nasbench201_1.1_mini_2.pt')

    print('-'*100)
    print("highest acc1 topologies")
    for i, r in enumerate(mini.get_all_sorted('acc1', 'cifar10-valid')):
        print('{:<20} {:<20} {}'.format(r.get('acc1', 'cifar10-valid'), str(r.arch_tuple), r.arch_str))
        if i > 8:
            break

    print('-'*100)
    c = 0
    print("highest acc1 topologies")
    for i, r in enumerate(mini.get_all_sorted('acc1', 'cifar10-valid')):
        if 1 not in r.arch_tuple:
            print('{:<20} {:<20} {}'.format(r.get('acc1', 'cifar10-valid'), str(r.arch_tuple), r.arch_str))
            c += 1
        if c > 9:
            break


if __name__ == '__main__':
    path_ = replace_standard_paths('{path_data}/nasbench201_1.1_mini.pt')
    # make_mini201('./nasbench2_1.1.pth', path_)
    explore(MiniNASBenchApi.load(path_))
