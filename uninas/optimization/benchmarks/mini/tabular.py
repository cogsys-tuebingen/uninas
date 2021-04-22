import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from typing import Union, Iterable
from uninas.optimization.benchmarks.mini.benchmark import MiniNASBenchmark
from uninas.optimization.benchmarks.mini.result import MiniResult
from uninas.optimization.hpo.uninas.values import ValueSpace, SpecificValueSpace
from uninas.utils.loggers.python import LoggerManager, Logger, log_in_columns, log_headline
from uninas.utils.args import Argument, Namespace
from uninas.utils.paths import replace_standard_paths
from uninas.register import Register


@Register.benchmark_set(mini=True, tabular=True)
class MiniNASTabularBenchmark(MiniNASBenchmark):
    """
    a tabular benchmark set
    """

    def __init__(self, default_data_set: Union[str, None], default_result_type: Union[str, None],
                 bench_name: str, bench_description: str,
                 value_space: ValueSpace, results: {int: MiniResult}, arch_to_idx: {str: int},
                 tuple_to_str: {tuple: str}, tuple_to_idx: {tuple: int}):
        super().__init__(default_data_set, default_result_type, value_space, bench_name, bench_description)
        self.results = results                  # {arc index -> result}
        self.arch_to_idx = arch_to_idx          # {arc str -> arc index}
        self.tuple_to_str = tuple_to_str        # {arc tuple -> arc str}
        self.tuple_to_idx = tuple_to_idx        # {arc tuple -> arc index}

    def subset(self, blacklist=(), other: MiniNASBenchmark = None, max_size=-1) -> 'MiniNASTabularBenchmark':
        """
        create a new benchmark as a subset of this one

        :param blacklist: tuple of indices, which arc choices to remove
        :param other: optional other benchmark, make sure to only keep architectures that are evaluated there as well
        :param max_size: maximum number of architectures to keep, all if <0
        """
        has_other = isinstance(other, MiniNASBenchmark)
        if (len(blacklist) == 0) and (max_size < 0) and (not has_other):
            return self

        # result subset
        remaining_results = []
        new_name = '%s SUBSET' % self.bench_name

        # blacklist
        for r in self._get_all():
            can_add = True
            for idx in blacklist:
                if idx in r.arch_tuple:
                    can_add = False
                    break
            if has_other and can_add:
                r2 = other.get_by_arch_tuple(r.arch_tuple)
                if r2 is None:
                    can_add = False
            if can_add:
                remaining_results.append(r)
        if len(blacklist) > 0:
            new_name = "%s%s" % (new_name, blacklist)

        # max results
        if len(remaining_results) > max_size > 0:
            rng = np.random.default_rng()
            remaining_results = rng.choice(remaining_results, size=max_size, replace=False)
            new_name = "%s size=%d" % (new_name, max_size)

        # update indices and reference dicts
        results, arch_to_idx, tuple_to_str, tuple_to_idx = {}, {}, {}, {}
        for i, r in enumerate(remaining_results):
            r.arch_index = i
            results[i] = r
            arch_to_idx[r.arch_str] = i
            tuple_to_str[r.arch_tuple] = r.arch_index
            tuple_to_idx[r.arch_tuple] = i

        # new value space
        value_space = deepcopy(self.value_space)
        for i in blacklist:
            value_space.remove_value(i)

        return self.__class__(default_data_set=self.default_data_set, default_result_type=self.default_result_type,
                              bench_name=new_name, bench_description=self.bench_description,
                              value_space=value_space, results=results, arch_to_idx=arch_to_idx,
                              tuple_to_str=tuple_to_str, tuple_to_idx=tuple_to_idx)

    def _save(self, save_path: str):
        if isinstance(save_path, str):
            save_path = replace_standard_paths(save_path)
            name = os.path.basename(save_path)
            path = os.path.dirname(save_path)
            os.makedirs(path, exist_ok=True)
            s = '%s/%s' % (path, name)
            data = dict(
                cls=self.__class__.__name__,
                default_data_set=self.default_data_set,
                default_result_type=self.default_result_type,
                value_space=self.get_value_space(),
                bench_name=self.bench_name,
                bench_description=self.bench_description,
                results={k: r.state_dict() for k, r in self.results.items()},
                arch_to_idx=self.arch_to_idx,
                tuple_to_str=self.tuple_to_str,
                tuple_to_idx=self.tuple_to_idx)
            torch.save(data, s)

    @classmethod
    def _load(cls, dct: dict) -> 'MiniNASTabularBenchmark':
        dct['results'] = {k: MiniResult(**v) for k, v in dct['results'].items()}
        return cls(**dct)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('path', default="{path_data}/bench.pt", type=str, is_path=True, help='path to the save file'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index: int = None) -> 'MiniNASTabularBenchmark':
        all_parsed = cls._all_parsed_arguments(args, index=index)
        path = all_parsed.pop('path')
        return cls.load(path, **all_parsed)

    @classmethod
    def merged(cls, benches: ['MiniNASTabularBenchmark'], merge_fun=np.mean):
        """ merge multiple benches into one, averaging their respective results """
        b0 = benches[0]
        all_results = []
        for bench in benches:
            all_results.extend(bench.results.values())
        merged_results = MiniResult.merge_result_list(all_results, merge_fun=merge_fun, ensure_same_size=True)

        # update indices and reference dicts
        results, arch_to_idx, tuple_to_str, tuple_to_idx = {}, {}, {}, {}
        for i, r in enumerate(merged_results):
            r.arch_index = i
            results[i] = r
            arch_to_idx[r.arch_str] = i
            tuple_to_str[r.arch_tuple] = r.arch_index
            tuple_to_idx[r.arch_tuple] = i

        return cls(default_data_set=b0.default_data_set, default_result_type=b0.default_result_type,
                   bench_name="Merged(%s)" % ", ".join([b.bench_name for b in benches]),
                   bench_description=b0.bench_description,
                   value_space=b0.value_space, results=results, arch_to_idx=arch_to_idx,
                   tuple_to_str=tuple_to_str, tuple_to_idx=tuple_to_idx)

    def get_specific_value_space(self) -> SpecificValueSpace:
        return SpecificValueSpace(self.get_all_architecture_tuples())

    def get_all_architecture_tuples(self) -> [tuple]:
        return [r.arch_tuple for r in self._get_all()]

    def _get_all(self) -> Iterable[MiniResult]:
        return self.results.values()

    def get_by_index(self, index: int) -> MiniResult:
        return self.results.get(index).set_defaults(self.default_data_set)

    def get_by_arch_str(self, arch_str: str) -> MiniResult:
        return self.results.get(self.arch_to_idx.get(arch_str)).set_defaults(self.default_data_set)

    def _get_by_arch_tuple(self, arch_tuple: tuple) -> Union[MiniResult, None]:
        arch_tuple = tuple([(v[0] if len(v) == 1 else tuple(v)) if isinstance(v, (tuple, list)) else v
                            for v in arch_tuple])  # flatten if possible
        return self.results.get(self.tuple_to_idx.get(arch_tuple))

    def size(self) -> int:
        return len(self.results)


def explore(mini: MiniNASTabularBenchmark, logger: Logger = None, n=-1, sort_by='acc1', maximize=True):
    if logger is None:
        logger = LoggerManager().get_logger()
    log_headline(logger, "highest acc1 topologies (%s, %s, %s)"
                 % (mini.get_name(), mini.get_default_data_set(), mini.get_default_result_type()))
    rows = [("%s rank" % sort_by, "acc1", "loss", "params", "flops", "latency", "tuple")]
    for i, r in enumerate(mini.get_all_sorted([sort_by], [maximize])):
        if i >= n > 0:
            break
        rows.append((i, r.get_acc1(), r.get_loss(), r.get_params(), r.get_flops(), r.get_latency(), r.arch_tuple))
    log_in_columns(logger, rows)


def plot(mini: MiniNASTabularBenchmark, keys: [str], maximize: [bool], add_pareto=True, legend=True):
    assert len(keys) == len(maximize) == 2
    all_entries = mini.get_all()
    plt.scatter([e.get(keys[0]) for e in all_entries], [e.get(keys[1]) for e in all_entries], label="all")
    if add_pareto:
        best_entries = mini.get_all_sorted(sorted_by=keys, maximize=maximize, only_best=True)
        plt.scatter([e.get(keys[0]) for e in best_entries], [e.get(keys[1]) for e in best_entries], label="best")
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])
    if legend:
        plt.legend()
    plt.show()
