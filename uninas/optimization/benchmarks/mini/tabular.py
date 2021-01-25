import os
import matplotlib.pyplot as plt
import torch
from typing import Union, Iterable
from uninas.optimization.benchmarks.mini.benchmark import MiniNASBenchmark
from uninas.optimization.benchmarks.mini.result import MiniResult
from uninas.optimization.hpo.uninas.values import ValueSpace
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
        self.results = results
        self.arch_to_idx = arch_to_idx
        self.tuple_to_str = tuple_to_str
        self.tuple_to_idx = tuple_to_idx

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
        default_data_set = cls._parsed_argument('default_data_set', args, index=index)
        default_result_type = cls._parsed_argument('default_result_type', args, index=index)
        path = cls._parsed_argument('path', args, index=index)
        ds = cls.load(path)
        if len(default_data_set) > 0:
            ds.set_default_data_set(default_data_set)
        if len(default_result_type) > 0:
            ds.set_default_result_type(default_result_type)
        return ds

    def get_all_architecture_tuples(self) -> [tuple]:
        return [r.arch_tuple for r in self._get_all()]

    def _get_all(self) -> Iterable[MiniResult]:
        return self.results.values()

    def get_by_index(self, index: int) -> MiniResult:
        return self.results.get(index).set_defaults(self.default_data_set)

    def get_by_arch_str(self, arch_str: str) -> MiniResult:
        return self.results.get(self.arch_to_idx.get(arch_str)).set_defaults(self.default_data_set)

    def _get_by_arch_tuple(self, arch_tuple: tuple) -> MiniResult:
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
