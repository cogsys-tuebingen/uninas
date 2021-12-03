from uninas.methods.abstract_method import AbstractMethod
from uninas.methods.strategy_manager import StrategyManager
from uninas.optimization.benchmarks.mini.tabular import MiniNASTabularBenchmark, explore
from uninas.optimization.benchmarks.mini.result import MiniResult
from uninas.optimization.hpo.uninas.population import Population
from uninas.utils.paths import replace_standard_paths
from uninas.register import Register
from uninas.builder import Builder


@Register.benchmark_set(mini=True, tabular=True)
class MiniNASSearchTabularBenchmark(MiniNASTabularBenchmark):
    """
    a tabular benchmark set created from evaluating a super-network
    """

    @classmethod
    def get_metric_keys(cls) -> [str]:
        return ['params', 'flops', 'latency', 'loss', 'acc1']

    @classmethod
    def make_from_population(cls, population: Population, method: AbstractMethod):
        """
        creating a mini bench dataset from an evaluated super-network
        """
        results = {}
        arch_to_idx = {}
        tuple_to_str = {}
        tuple_to_idx = {}

        space = StrategyManager().get_value_space(unique=True)
        data_set_name = method.get_data_set().__class__.__name__
        space_name = method.get_network().get_model_name()
        default_result_type = "test"

        for i, candidate in enumerate(population.get_candidates()):
            # first use all estimated metrics
            # if they contain e.g. "acc1/valid", create a sub dict
            metrics = {}
            for k, v in candidate.metrics.items():
                splits = k.split('/')
                if len(splits) == 1:
                    metrics[splits[0]] = {data_set_name: v}
                else:
                    metrics[splits[0]] = metrics.get(splits[0], {})
                    metrics[splits[0]][data_set_name] = metrics[splits[0]].get(data_set_name, {})
                    metrics[splits[0]][data_set_name][splits[1]] = v
                    default_result_type = splits[1]
            # now make sure all keys exist
            for k in MiniResult.get_metric_keys():
                metrics[k] = metrics.get(k, {data_set_name: -1})
            # result
            r = MiniResult(
                arch_index=i,
                arch_str="%s(%s)" % (space_name, ", ".join([str(v) for v in candidate.values])),
                arch_tuple=candidate.values,
                **metrics
            )

            assert tuple_to_str.get(r.arch_tuple) is None, "can not yet merge duplicate architecture results"
            results[i] = r
            arch_to_idx[r.arch_str] = i
            tuple_to_idx[r.arch_tuple] = i
            tuple_to_str[r.arch_tuple] = r.arch_str

        data_sets = list(results.get(0).params.keys())
        return MiniNASSearchTabularBenchmark(
            default_data_set=data_sets[0],
            default_result_type=default_result_type,
            bench_name="%s on %s" % (space_name, data_sets[0]),
            bench_description="super-network evaluation results",
            value_space=space, results=results, arch_to_idx=arch_to_idx,
            tuple_to_str=tuple_to_str, tuple_to_idx=tuple_to_idx)


if __name__ == '__main__':
    Builder()
    path_ = replace_standard_paths('{path_tmp}/s2_bench/bench-SearchUninasNetwork_on_Imagenet1000Data.pt')
    mini_ = MiniNASSearchTabularBenchmark.load(path_)
    explore(mini_)
