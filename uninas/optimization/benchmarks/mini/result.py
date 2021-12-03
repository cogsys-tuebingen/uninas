from typing import Tuple
from collections import defaultdict
import numpy as np


class MiniResult:
    """
    a result that contains only the latest loss/acc1,
    all values are averaged over multiple runs/seeds by the topology
    """

    _type_keys = ['train', 'valid', 'test']

    def __init__(self, arch_index: int, arch_str: str, arch_tuple: Tuple[int],
                 params: {str: int}, flops: {str: float}, latency: {str: float},
                 loss: {str: {str: float}}, acc1: {str: {str: float}},
                 default_data_set: str = None,
                 default_result_type: str = 'test'):
        """
        :param arch_index: id of this architecture
        :param arch_str: id of this architecture
        :param arch_tuple: id of this architecture
        :param params: {data set: num params}
        :param flops: {data set: num flops}
        :param latency: {data set: latency}
        :param loss: {data set: {result_type: loss}}   result_type is in {'train', 'valid', 'test'}
        :param acc1: {data set: {result_type: acc1}}
        :param default_data_set: return values of this data set, unless requested otherwise
        :param default_result_type: return values of train/valid/test
        """

        # make sure that the result_type strings are as expected
        for dct1 in [loss, acc1]:
            for k1, dct2 in dct1.items():
                if isinstance(dct2, dict):
                    for k3 in dct2.keys():
                        assert k3 in self._type_keys
                    for k3 in self._type_keys:
                        dct1[k1][k3] = dct1[k1].get(k3, -1)
                else:
                    v = dct2
                    dct1[k1] = {}
                    for k3 in self._type_keys:
                        dct1[k1][k3] = v

        self.arch_index = arch_index
        self.arch_str = arch_str
        self.arch_tuple = arch_tuple
        self.params = params
        self.flops = flops
        self.latency = latency
        self.loss = loss
        self.acc1 = acc1

        # defaults
        self.default_data_set = default_data_set
        self.default_result_type = default_result_type

    @classmethod
    def get_metric_keys(cls) -> [str]:
        return ['params', 'flops', 'latency', 'loss', 'acc1']

    def set_defaults(self, default_data_set: str = None, default_result_type: str = None) -> 'MiniResult':
        """
        set defaults
        """
        self.default_data_set = default_data_set
        if isinstance(default_result_type, str):
            self.default_result_type = default_result_type
        return self

    def get_data_sets(self) -> [str]:
        return list(self.params.keys())

    def _get_defaults(self, data_set: str = None, result_type: str = None) -> (str, str):
        ds, rt = self.default_data_set, self.default_result_type
        if isinstance(data_set, str):
            ds = data_set
        if isinstance(result_type, str):
            rt = result_type
        if (ds is None) or (rt is None):
            raise ValueError("either data_set or result_type is None (default and given), [ds=%s, rt=%s]" % (ds, rt))
        return ds, rt

    def state_dict(self) -> dict:
        dct = dict()
        dct.update(dict(
            arch_index=self.arch_index,
            arch_str=self.arch_str,
            arch_tuple=self.arch_tuple,
        ))
        dct.update(dict(
            params=self.params,
            flops=self.flops,
            latency=self.latency,
            loss=self.loss,
            acc1=self.acc1,
        ))
        return dct

    def get_info_str(self, data_set: str = None, result_type: str = None) -> str:
        data_set, result_type = self._get_defaults(data_set, result_type)

        return "{cls}(index={idx}, acc1={acc1}, loss={loss}, params={par}, flops={flops}, latency={lat})".format(**{
            'cls': self.__class__.__name__,
            'idx': str(self.arch_index),
            'acc1': str(self.acc1.get(data_set, {}).get(result_type)),
            'loss': str(self.loss.get(data_set, {}).get(result_type)),
            'par': str(self.params.get(data_set)),
            'flops': str(self.flops.get(data_set)),
            'lat': str(self.latency.get(data_set)),
        })

    def get_log_dict(self) -> {str: float}:
        log_dict = {}

        # loss and acc1
        for name, dct in [('loss', self.loss), ('acc1', self.acc1)]:
            for ds, dct2 in dct.items():
                for rt, v in dct2.items():
                    log_dict['%s/%s/%s' % (name, ds, rt)] = v

        # other metrics
        for name, dct in [('flops', self.flops), ('params', self.params), ('latency', self.latency)]:
            for ds, v in dct.items():
                log_dict['%s/%s' % (name, ds)] = v
        return log_dict

    def get(self, kind: str, data_set: str = None, result_type: str = None) -> float:
        fun = {
            'params': self.get_params,
            'flops': self.get_flops,
            'latency': self.get_latency,
            'loss': self.get_loss,
            'acc1': self.get_acc1,
        }[kind]
        return float(fun(data_set, result_type))

    def get_params(self, data_set: str = None, result_type: str = None) -> int:
        data_set, result_type = self._get_defaults(data_set, result_type)
        return self.params.get(data_set, -1)

    def get_flops(self, data_set: str = None, result_type: str = None) -> float:
        data_set, result_type = self._get_defaults(data_set, result_type)
        return self.flops.get(data_set, -1.0)

    def get_latency(self, data_set: str = None, result_type: str = None) -> float:
        data_set, result_type = self._get_defaults(data_set, result_type)
        return self.latency.get(data_set, -1.0)

    def get_loss(self, data_set: str = None, result_type: str = None) -> float:
        data_set, result_type = self._get_defaults(data_set, result_type)
        return self.loss.get(data_set, {}).get(result_type, -1.0)

    def get_acc1(self, data_set: str = None, result_type: str = None) -> float:
        data_set, result_type = self._get_defaults(data_set, result_type)
        return self.acc1.get(data_set, {}).get(result_type, -1.0)

    def print(self, print_fun=print, prefix=''):
        for n, v in self.state_dict().items():
            print_fun('{}{:<13}{:}'.format(prefix, n, v))

    @classmethod
    def merge_results(cls, results: ['MiniResult'], merge_fun=np.mean) -> 'MiniResult':
        """ merge results """

        def merge(items: list):
            # make sure they have the same type, and are not None
            assert items[0] is not None
            t = type(items[0])
            assert all([type(itm) == t for itm in items])

            # if they are dicts, recursion
            if isinstance(items[0], dict):
                # make sure all have exactly the same keys
                keys = list(items[0].keys())
                for item in items:
                    assert len(list(item.keys())) == len(keys)
                    for key in keys:
                        assert item.get(key, None) is not None

                # merge values
                new_dict = {}
                for key in keys:
                    new_dict[key] = merge([item.get(key) for item in items])
                return new_dict

            # otherwise make sure either all or none have values > 0
            else:
                assert all([i < 0 for i in items]) or all([i >= 0 for i in items])
                return merge_fun(items)

        r0 = results[0]
        return cls(
            arch_index=r0.arch_index, arch_str=r0.arch_str, arch_tuple=r0.arch_tuple,
            params=merge([r.params for r in results]),
            flops=merge([r.flops for r in results]),
            latency=merge([r.latency for r in results]),
            loss=merge([r.loss for r in results]),
            acc1=merge([r.acc1 for r in results]),
            default_data_set=r0.default_data_set,
            default_result_type=r0.default_result_type)

    @classmethod
    def merge_result_list(cls, results: ['MiniResult'], merge_fun=np.mean, ensure_same_size=True) -> ['MiniResult']:
        """
        merge results by their architecture tuple

        :param results: all results to merge
        :param merge_fun: how to merge values
        :param ensure_same_size: assert that each arc tuple has the same number of associated results
        """
        # cluster by arc tuple
        clusters = defaultdict(list)
        for r in results:
            clusters[r.arch_tuple].append(r)
        if ensure_same_size:
            lengths = [len(v) for v in clusters.values()]
            assert min(lengths) == max(lengths), "Different architectures have different number of results!"

        # merge each cluster, give new arc indices
        merged_results = []
        for i, cluster in enumerate(clusters.values()):
            r = cls.merge_results(cluster, merge_fun=merge_fun)
            r.arch_index = i
            merged_results.append(r)

        return merged_results
