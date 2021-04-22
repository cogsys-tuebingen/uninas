from typing import Union, Iterable
from uninas.optimization.benchmarks.mini.benchmark import MiniNASBenchmark
from uninas.optimization.benchmarks.mini.result import MiniResult
from uninas.optimization.hpo.uninas.values import ValueSpace
from uninas.utils.args import Namespace


class MiniNASSurrogateBenchmark(MiniNASBenchmark):
    """
    a surrogate benchmark set
    """

    def __init__(self, default_data_set: Union[str, None], default_result_type: Union[str, None],
                 bench_name: str, bench_description: str, value_space: ValueSpace):
        super().__init__(default_data_set, default_result_type, value_space, bench_name, bench_description)

    def _save(self, save_path: str):
        raise NotImplementedError

    @classmethod
    def _load(cls, dct: dict) -> 'MiniNASSurrogateBenchmark':
        raise NotImplementedError

    @classmethod
    def from_args(cls, args: Namespace, index: int = None) -> 'MiniNASSurrogateBenchmark':
        raise NotImplementedError

    def _get_all(self) -> Iterable[MiniResult]:
        raise NotImplementedError

    def _get_by_arch_tuple(self, arch_tuple: tuple) -> Union[MiniResult, None]:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError
