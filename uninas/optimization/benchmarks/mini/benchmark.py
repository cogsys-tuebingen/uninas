import os
import numpy as np
import torch
from typing import Union, Iterable
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from uninas.optimization.benchmarks.mini.result import MiniResult
from uninas.optimization.hpo.uninas.values import ValueSpace, SpecificValueSpace
from uninas.utils.args import ArgsInterface, Namespace, Argument
from uninas.utils.loggers.python import LoggerManager
from uninas.utils.paths import maybe_download
from uninas.register import Register


class MiniNASBenchmark(ArgsInterface):

    def __init__(self, default_data_set: Union[str, None], default_result_type: Union[str, None],
                 value_space: ValueSpace, bench_name: str, bench_description: str, tmp_name: str = ""):
        super().__init__()
        self.default_data_set = default_data_set
        self.default_result_type = default_result_type

        self.value_space = value_space
        self.bench_name = bench_name
        self.bench_description = bench_description
        self.tmp_name = tmp_name

    def get_name(self) -> str:
        if len(self.tmp_name) > 0:
            return self.tmp_name
        return self.bench_name

    def set_default_data_set(self, default_data_set: str = None) -> 'MiniNASBenchmark':
        """ set the default data set """
        self.default_data_set = default_data_set
        return self

    def get_default_data_set(self) -> str:
        """ get the current default data set, empty string if not set """
        return self.default_data_set if isinstance(self.default_data_set, str) else ''

    def set_default_result_type(self, default_result_type: str = None) -> 'MiniNASBenchmark':
        """ set the default result type (train/valid/test) """
        self.default_result_type = default_result_type
        return self

    def get_default_result_type(self) -> str:
        """ get the current default result type, empty string if not set """
        return self.default_result_type if isinstance(self.default_result_type, str) else ''

    def save_in_dir(self, save_dir: str):
        self.save("%s/bench-%s.pt" % (save_dir, self.get_name().replace(' ', '_')))

    def save(self, save_path: str):
        self._save(save_path)
        LoggerManager().get_logger().info("Saved %s to %s" % (self.__class__.__name__, save_path))

    def _save(self, save_path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path_or_url: str, default_data_set="", default_result_type="", tmp_name="") -> 'MiniNASBenchmark':
        """ load the benchmark from a file, automatically finds the correct class to load """
        # find file and used benchmark class
        path = maybe_download(path_or_url)
        assert path is not None, "path is None for: %s" % path_or_url
        assert os.path.exists(path), "does not exist: %s" % path
        assert os.path.isfile(path), "not a file: %s" % path
        dct = torch.load(path)
        cls_ = Register.benchmark_sets.get(dct.pop('cls'))
        assert issubclass(cls_, MiniNASBenchmark)
        # load benchmark, set optional defaults
        ds = cls_._load(dct)
        if len(default_data_set) > 0:
            ds.set_default_data_set(default_data_set)
        if len(default_result_type) > 0:
            ds.set_default_result_type(default_result_type)
        ds.tmp_name = tmp_name
        return ds

    @classmethod
    def _load(cls, dct: dict) -> 'MiniNASBenchmark':
        raise NotImplementedError

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('default_data_set', default="", type=str, help='default data set to use, ignored if empty'),
            Argument('default_result_type', default="test", type=str, choices=['train', 'valid', 'test'],
                     help='default result type to use'),
            Argument('tmp_name', default="", type=str, help='change the name of the benchmark, useful for plotting'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index: int = None) -> 'MiniNASBenchmark':
        raise NotImplementedError

    def get_value_space(self) -> ValueSpace:
        return self.value_space

    def get_specific_value_space(self) -> SpecificValueSpace:
        raise NotImplementedError

    def get_my_space_lower_upper(self) -> (np.array, np.array):
        """ lower and upper limits (both included) for pymoo """
        space = self.get_value_space()
        lower, upper = [], []
        for s in space.get_values():
            lower.append(s.get_min_value())
            upper.append(s.get_max_value())
        return np.array(lower), np.array(upper)

    def get_all(self) -> [MiniResult]:
        return [r.set_defaults(self.default_data_set, self.default_result_type) for r in self._get_all()]

    def _get_all(self) -> Iterable[MiniResult]:
        raise NotImplementedError

    def get_some(self, n=5) -> Iterable[MiniResult]:
        """ get any results """
        for i, r in enumerate(self._get_all()):
            if i >= n:
                break
            yield r.set_defaults(self.default_data_set, self.default_result_type)

    def get_all_datasets(self) -> [str]:
        for r in self._get_all():
            return r.get_data_sets()

    def get_all_sorted(self, sorted_by: [str], maximize: [bool], data_set: str = None, only_best=False) -> [MiniResult]:
        """
        get all / the best entries sorted by 1 to N objectives

        :param sorted_by: sort by 'acc1', 'flops', ... single or multi-objective
        :param maximize: for each sorted_by key, whether to maximize the value
        :param data_set: use a specific data set, otherwise default
        :param only_best: return all or only the pareto front
        """
        assert len(sorted_by) == len(maximize) > 0
        # only one objective
        if len(sorted_by) == 1:
            all_entries = sorted(self.get_all(), key=lambda s: s.get(sorted_by[0], data_set), reverse=maximize[0])
            if only_best:
                i, best_entries = 1, [all_entries[0]]
                while all_entries[0].get(sorted_by[0]) == all_entries[i].get(sorted_by[0]):
                    i += 1
                    best_entries.append(all_entries[i])
                all_entries = best_entries
            return all_entries
        # multiple objectives
        else:
            nds = NonDominatedSorting()
            # first get all values, minimization problem
            entries = self.get_all()
            all_values = np.zeros(shape=(len(entries), len(sorted_by)))
            for i, entry in enumerate(entries):
                all_values[i] = [entry.get(s, data_set) * (-1 if m else 1)
                                 for s, m in zip(sorted_by, maximize)]
            # if we want only the best rank, we can sort and concat smaller groups for speed
            if only_best:
                # find best in small subgroups
                group_size, best_idx = 200, []
                for i in range(0, len(all_values) // group_size + 1):
                    start, end = group_size * i, min(group_size * (i + 1), len(all_values))
                    idx = np.arange(start, end)
                    group = all_values[idx]
                    if group.shape[0] > 0:
                        pareto_idx = nds.do(group, only_non_dominated_front=True) + start
                        best_idx.append(pareto_idx)
                # concat subgroups and find best
                best_idx = np.concatenate(best_idx, axis=0)
                pareto_idx = nds.do(all_values[best_idx], only_non_dominated_front=True)
                best_idx = best_idx[pareto_idx]
                all_entries = [entries[i] for i in best_idx]
            # we want all, sorted
            else:
                sorted_idx = nds.do(all_values, only_non_dominated_front=True)
                all_entries = [entries[i] for i in sorted_idx]
        return sorted(all_entries, key=lambda s: s.get(sorted_by[0], data_set), reverse=maximize[0])

    def get_by_arch_tuple(self, arch_tuple: tuple) -> Union[MiniResult, None]:
        result = self._get_by_arch_tuple(arch_tuple)
        if isinstance(result, MiniResult):
            result.set_defaults(self.default_data_set, self.default_result_type)
        return result

    def _get_by_arch_tuple(self, arch_tuple: tuple) -> Union[MiniResult, None]:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def print_info(self, print_fun=print):
        print_fun('{cls}(name="{name}", num_results={nr})'.format(**{
            'cls': self.__class__.__name__,
            'name': self.bench_name,
            'nr': self.size(),
        }))

    def print_some(self, n=5, print_fun=print):
        for r in self.get_some(n=n):
            print_fun('-'*180)
            r.print(print_fun=print_fun)

    def get_result_dict(self, result: MiniResult) -> {str: float}:
        log_dct = dict()
        for k, v in result.get_log_dict().items():
            log_dct['bench/%s/%s' % (self.bench_name, k)] = v
        return log_dct

    def is_tabular(self) -> bool:
        return Register.get_my_kwargs(self.__class__).get('tabular', False)
