from collections import defaultdict
from uninas.optimization.benchmarks.mini.tabular import MiniNASTabularBenchmark, plot
from uninas.optimization.benchmarks.mini.result import MiniResult
from uninas.optimization.hpo.uninas.values import DiscreteValues, ValueSpace
from uninas.modules.primitives.bench201 import Bench201Primitives
from uninas.utils.paths import replace_standard_paths
from uninas.utils.loggers.python import LoggerManager, log_in_columns, log_headline
from uninas.register import Register
from uninas.builder import Builder


try:
    from nats_bench import create, NATStopology


    @Register.benchmark_set(mini=True, tabular=True)
    class MiniNATSBenchTabularBenchmark(MiniNASTabularBenchmark):
        """
        https://arxiv.org/abs/2009.00437
        https://github.com/D-X-Y/NATS-Bench

        the most relevant parts of the original API, requires significantly less RAM
        """
        _arc_str_to_idx = {k: i for i, k in enumerate(Bench201Primitives.get_order())}

        @classmethod
        def get_arc_tuples_from_str(cls, arch_str: str) -> (tuple, tuple):
            """
            get the architecture tuples, [str] and [idx]
            :param arch_str:
            :return:
            """
            ops_str = arch_str[1:-1].replace('|+', '').split('|')
            ops_str = [s.split('~')[0] for s in ops_str]
            return ops_str, tuple([int(cls._arc_str_to_idx.get(s)) for s in ops_str])

        @classmethod
        def make_from_full_api(cls, api: NATStopology):
            """ creating a mini bench dataset from NAS-Bench-201 data """
            data_sets = ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']
            hp = '200'
            results = {}
            arch_to_idx = {}
            tuple_to_str = {}
            tuple_to_idx = {}

            for i, arch_str in enumerate(api.meta_archs):
                # name to tuple, tuple to name and index
                ops_str, ops_tuple = cls.get_arc_tuples_from_str(arch_str)
                tuple_to_str[ops_tuple] = ops_str
                tuple_to_idx[ops_tuple] = i
                arch_to_idx[arch_str] = i

                # cost stats
                params = defaultdict(list)
                flops = defaultdict(list)
                latency = defaultdict(list)

                # training stats
                loss = defaultdict(lambda: defaultdict(list))
                acc1 = defaultdict(lambda: defaultdict(list))

                # update stats per used data set
                for ds in data_sets:
                    info_res = api.get_more_info(i, ds, iepoch=None, hp=hp, is_random=False)
                    info_cost = api.get_cost_info(i, ds)

                    params[ds].append(info_cost.get('params', -1))
                    flops[ds].append(info_cost.get('flops', -1))
                    latency[ds].append(info_cost.get('latency', -1))

                    loss[ds]['train'].append(info_res.get('train-loss', -1))
                    loss[ds]['valid'].append(info_res.get('valid-loss', -1))
                    loss[ds]['test'].append(info_res.get('test-loss', -1))

                    acc1[ds]['train'].append(info_res.get('train-accuracy', -1))
                    acc1[ds]['valid'].append(info_res.get('valid-accuracy', -1))
                    acc1[ds]['test'].append(info_res.get('test-accuracy', -1))

                results[i] = MiniResult(
                    arch_index=i,
                    arch_str=arch_str,
                    arch_tuple=ops_tuple,
                    params={k: sum(v) / len(v) for k, v in params.items()},
                    flops={k: sum(v) / len(v) for k, v in flops.items()},
                    latency={k: sum(v) / len(v) for k, v in latency.items()},
                    loss={k1: {k2: sum(v2) / len(v2) for k2, v2 in v1.items()} for k1, v1 in loss.items()},
                    acc1={k1: {k2: sum(v2) / len(v2) for k2, v2 in v1.items()} for k1, v1 in acc1.items()},
                )

                # keep memory in check
                api.arch2infos_dict.clear()

            return cls(
                default_data_set="cifar10", default_result_type='test', bench_name="NATS-BENCH mini",
                bench_description="mini variant of NATS-BENCH that only contains final results",
                value_space=ValueSpace(*[DiscreteValues.interval(0, 5) for _ in range(6)]),
                results=results, arch_to_idx=arch_to_idx, tuple_to_str=tuple_to_str, tuple_to_idx=tuple_to_idx)


    def make_nats(path_full: str, path_save: str = None) -> MiniNATSBenchTabularBenchmark:
        api = create(replace_standard_paths(path_full), 'tss', fast_mode=True, verbose=True)
        mini = MiniNATSBenchTabularBenchmark.make_from_full_api(api)
        if isinstance(path_save, str):
            mini.save(path_save)
        return mini


    def explore(mini: MiniNATSBenchTabularBenchmark):
        logger = LoggerManager().get_logger()

        # some stats of specific results
        logger.info('\n'*5)
        logger.info("%s - %s" % (mini.get_name(), mini.size()))
        logger.info(mini.get_by_arch_tuple((4, 3, 2, 1, 0, 2)).get_info_str('cifar10'))
        logger.info("")
        mini.get_by_arch_tuple((1, 2, 1, 2, 3, 4)).print(logger.info)
        logger.info("")
        mini.get_by_index(1554).print(logger.info)
        logger.info("")
        mini.get_by_arch_str('|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|').print(logger.info)
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
        mini2 = mini.subset(blacklist=(1,))
        rows = [("acc1", "params", "arch tuple", "arch str")]
        log_headline(logger, "highest acc1 topologies without skip (%s, %s, %s)" % (mini2.get_name(), mini2.get_default_data_set(), mini2.get_default_result_type()))
        for i, r in enumerate(mini2.get_all_sorted(['acc1'], [True])):
            rows.append((r.get_acc1(), r.get_params(), str(r.arch_tuple), r.arch_str))
            if i > 8:
                break
        log_in_columns(logger, rows)


    if __name__ == '__main__':
        Builder()
        path_ = '{path_data}/bench/nats/nats_bench_1.1_mini_m%s_%s.pt'
        # path_ = "https://cloud.cs.uni-tuebingen.de/index.php/s/kaxGB7csptYE9Jt/download"  # not up to date
        # mini_ = make_nats('{path_data}/NATS-tss-v1_0-3ffb9-simple/', path_ % ('', 'all'))
        mini_ = MiniNATSBenchTabularBenchmark.load(path_ % ('', 'all'))

        # subsets for experiments
        for subset in [(), (0,), (0, 1, 4)]:
            m = mini_.subset(blacklist=subset, max_size=1000)
            # m.save(path_ % (''.join([str(s) for s in subset]), '1k'))
            # explore(m)
            plot(m, ['flops', 'acc1'], [False, True], add_pareto=True)

        # viz
        mini_.set_default_result_type('test')  # train, valid, test
        mini_.set_default_data_set('cifar10')  # cifar10, cifar10-valid, cifar100, ImageNet16-120
        explore(mini_)
        plot(mini_, ['flops', 'acc1'], [False, True], add_pareto=True)


except ImportError as e:
    Register.missing_import(e)
