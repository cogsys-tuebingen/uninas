import json
from uninas.optimization.benchmarks.mini.tabular import MiniNASTabularBenchmark, plot
from uninas.optimization.benchmarks.mini.result import MiniResult
from uninas.optimization.hpo.uninas.values import DiscreteValues, ValueSpace
from uninas.utils.paths import replace_standard_paths
from uninas.utils.loggers.python import LoggerManager, log_in_columns, log_headline
from uninas.register import Register
from uninas.builder import Builder


@Register.benchmark_set(mini=True, tabular=True)
class MiniNASBenchMacroTabularBenchmark(MiniNASTabularBenchmark):
    """
    https://arxiv.org/abs/2103.11922
    https://github.com/xiusu/NAS-Bench-Macro
    """

    @classmethod
    def make_from_json(cls, json_path: str):
        """ creating a mini bench dataset from the json file """
        with open(json_path) as json_file:
            json_data = json.load(json_file)

        ds = 'cifar10'
        results = {}
        arch_to_idx = {}
        tuple_to_str = {}
        tuple_to_idx = {}

        for i, (arch_str, arch_info) in enumerate(json_data.items()):
            # name to tuple, tuple to name and index
            ops_str, ops_tuple = arch_str, tuple([int(s) for s in arch_str])
            tuple_to_str[ops_tuple] = ops_str
            tuple_to_idx[ops_tuple] = i
            arch_to_idx[arch_str] = i

            params = {ds: arch_info.get('params', -1)}
            flops = {ds: arch_info.get('flops', -1)}
            latency = {ds: arch_info.get('latency', -1)}  # does not exist

            loss = {ds: {}}  # does not exist
            acc1 = {ds: {'test': arch_info.get('mean_acc', -1)}}

            results[i] = MiniResult(
                arch_index=i,
                arch_str=arch_str,
                arch_tuple=ops_tuple,
                params=params,
                flops=flops,
                latency=latency,
                loss=loss,
                acc1=acc1,
            )

        return cls(
            default_data_set="cifar10", default_result_type='test', bench_name="NAS-Bench-Macro mini",
            bench_description="NAS-Bench-Macro that only contains final results",
            value_space=ValueSpace(*[DiscreteValues.interval(0, 3) for _ in range(8)]),
            results=results, arch_to_idx=arch_to_idx, tuple_to_str=tuple_to_str, tuple_to_idx=tuple_to_idx)


def make_nbm(json_path: str, path_save: str = None) -> MiniNASBenchMacroTabularBenchmark:
    mini = MiniNASBenchMacroTabularBenchmark.make_from_json(json_path)
    if isinstance(path_save, str):
        mini.save(path_save)
    return mini


def explore(mini: MiniNASBenchMacroTabularBenchmark):
    logger = LoggerManager().get_logger()

    # some stats of specific results
    logger.info(mini.get_by_arch_tuple((1, 2, 1, 2, 0, 2, 0, 1)).get_info_str('cifar10'))
    logger.info("")
    mini.get_by_arch_tuple((1, 2, 1, 2, 0, 2, 0, 1)).print(logger.info)
    logger.info("")
    mini.get_by_index(1554).print(logger.info)
    logger.info("")
    mini.get_by_arch_str('00122102').print(logger.info)
    logger.info("")

    # best results by acc1
    rows = [("acc1", "params", "arch tuple", "arch str")]
    log_headline(logger, "highest acc1 topologies (%s, %s, %s)" % (mini.get_name(), mini.get_default_data_set(), mini.get_default_result_type()))
    for i, r in enumerate(mini.get_all_sorted(['acc1'], [True])):
        rows.append((r.get_acc1(), r.get_params(), str(r.arch_tuple), r.arch_str))
        if i > 8:
            break
    log_in_columns(logger, rows)
    logger.info("")

    # best results by acc1, subset
    mini2 = mini.subset(blacklist=(0,))
    rows = [("acc1", "params", "arch tuple", "arch str")]
    log_headline(logger, "highest acc1 topologies without skip (%s, %s, %s)" % (mini2.get_name(), mini2.get_default_data_set(), mini2.get_default_result_type()))
    for i, r in enumerate(mini2.get_all_sorted(['acc1'], [True])):
        rows.append((r.get_acc1(), r.get_params(), str(r.arch_tuple), r.arch_str))
        if i > 8:
            break
    log_in_columns(logger, rows)


if __name__ == '__main__':
    Builder()
    path_ = '{path_data}/bench/nasbench_macro/nasbench_macro_mini_m_%s.pt'
    mini_ = make_nbm(replace_standard_paths('~/CodeGit/NAS-Bench-Macro/data/nas-bench-macro_cifar10.json'), path_ % 'all')
    # mini_ = MiniNASBenchMacroTabularBenchmark.load(path_ % 'all')
    # mini_.subset(max_size=1000).save(path_ % '1k')

    mini_.set_default_result_type('test')  # train, valid, test
    explore(mini_)

    plot(mini_, ['flops', 'acc1'], [False, True], add_pareto=True)
