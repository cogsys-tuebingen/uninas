from collections import defaultdict
from uninas.modules.primitives.bench101 import Bench101Primitives
from uninas.optimization.benchmarks.mini.tabular import MiniNASTabularBenchmark, plot, explore
from uninas.optimization.benchmarks.mini.result import MiniResult
from uninas.optimization.hpo.uninas.values import DiscreteValues, ValueSpace
from uninas.utils.paths import replace_standard_paths
from uninas.utils.loggers.python import LoggerManager, log_in_columns, log_headline
from uninas.register import Register
from uninas.builder import Builder


try:
    from nasbench import api


    @Register.benchmark_set(mini=True, tabular=True)
    class MiniNASBench101TabularBenchmark(MiniNASTabularBenchmark):
        """
        https://arxiv.org/abs/1902.09635
        https://github.com/google-research/nasbench
        """

        _op_names = ['input']

        @classmethod
        def make_from_full_api(cls, path: str, num_ops: int = 5):
            """
            creating a mini bench dataset from NAS-Bench-201 data

            :param path: path of the tfrecords file
            :param num_ops: only consider architectures with this many operations (input/output not counted)
            """
            assert 0 < num_ops < 6
            bench = api.NASBench(replace_standard_paths(path))
            data_set = 'cifar10'

            str_to_idx = {k: i for i, k in enumerate(Bench101Primitives.get_order())}
            results = {}
            arch_to_idx = {}
            tuple_to_str = {}
            tuple_to_idx = {}

            for i, (arc_hash, fixed_arc_stats) in enumerate(bench.fixed_statistics.items()):
                arch_lst = fixed_arc_stats["module_operations"][1:-1]  # remove input/output

                if len(arch_lst) != num_ops:
                    continue

                arch_str = "ops=%s, adjacancy=%s, hash=%s"\
                           % (fixed_arc_stats["module_operations"], fixed_arc_stats["module_adjacency"], arc_hash)
                arch_to_idx[arch_str] = i
                ops_tuple = tuple([int(str_to_idx.get(s)) for s in arch_lst])
                tuple_to_str[ops_tuple] = arch_str
                tuple_to_idx[ops_tuple] = i

                # query the result from bench101
                model_spec = api.ModelSpec(
                    matrix=fixed_arc_stats['module_adjacency'],
                    ops=fixed_arc_stats['module_operations'])
                data = bench.query(model_spec)

                # TODO tuple highly redundant... also needs matrix
                raise NotImplementedError("parsing not yet implemented, not certain how to do so best")

                """
                how to do that?
                (op0, adj0, op1, adj1, ...) ?
                    bad for input...
                """

                results[i] = MiniResult(
                    arch_index=i,
                    arch_str=str(data['module_operations']),
                    arch_tuple=ops_tuple,
                    params={data_set: data['trainable_parameters']},
                    flops={data_set: -1},
                    latency={data_set: -1},
                    loss={data_set: -1},
                    acc1={data_set: {
                        'train': data['train_accuracy'],
                        'valid': data['validation_accuracy'],
                        'test': data['test_accuracy'],
                    }},
                )

            return cls(
                default_data_set="cifar10", default_result_type='test',
                bench_name="NASBENCH 101 mini for architectures with %d operations (+ input/output)" % num_ops,
                bench_description="mini variant of NASBENCH 101 that only contains final results",
                value_space=ValueSpace(*[DiscreteValues.interval(0, 3) for _ in range(5)]),
                results=results, arch_to_idx=arch_to_idx, tuple_to_str=tuple_to_str, tuple_to_idx=tuple_to_idx)


    def make_nasbench(path_full: str, path_save: str = None, num_ops: int = 5) -> MiniNASBench101TabularBenchmark:
        mini = MiniNASBench101TabularBenchmark.make_from_full_api(path_full, num_ops)
        if isinstance(path_save, str):
            mini.save(path_save)
        return mini


    if __name__ == '__main__':
        Builder()
        path_ = '{path_data}/nasbench101_5ops_mini.pt'
        mini_ = make_nasbench('{path_data}/nasbench_only108.tfrecord', path_, num_ops=5)
        # mini_ = MiniNASBench101TabularBenchmark.load(path_)

        mini_.set_default_result_type('valid')  # train, valid, test
        mini_.set_default_data_set('cifar100')  # cifar10, cifar10-valid, cifar100, ImageNet16-120
        explore(mini_)

        plot(mini_, ['flops', 'acc1'], [False, True], add_pareto=True)

        # mini_.save('{path_data}/nasbench201_1.1_mini_2.pt')
        # mini_.save(path_)


except ImportError as e:
    Register.missing_import(e)

except AttributeError as e:
    # not using TF 1.x
    pass
