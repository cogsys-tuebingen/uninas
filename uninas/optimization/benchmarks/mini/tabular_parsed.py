import os
import json
import numpy as np
from uninas.optimization.benchmarks.mini.tabular import MiniNASTabularBenchmark, explore, plot
from uninas.optimization.benchmarks.mini.result import MiniResult
from uninas.optimization.hpo.uninas.values import DiscreteValues, ValueSpace
from uninas.utils.paths import replace_standard_paths, name_task_config, find_all_files
from uninas.utils.parsing.tensorboard_ import find_tb_files, read_event_files
from uninas.utils.misc import split
from uninas.utils.torch.standalone import get_dataset_from_json, get_network
from uninas.register import Register
from uninas.builder import Builder


@Register.benchmark_set(mini=True, tabular=True)
class MiniNASParsedTabularBenchmark(MiniNASTabularBenchmark):
    """
    go through a directory of training save-dirs,
    parsing the results of each single training run and its used architecture

    assumptions:
        - all networks are created as RetrainFromSearchUninasNetwork, so that their genes can be easily read
    """

    @classmethod
    def make_from_single_dir(cls, path: str, space_name: str, arch_index: int) -> MiniResult:
        """
        creating a mini result by parsing a training process
        """

        # find gene and dataset in the task config
        task_configs = find_all_files(path, extension=name_task_config)
        assert len(task_configs) == 1

        with open(task_configs[0]) as config_file:
            config = json.load(config_file)
            gene = config.get('{cls_network}.gene')
            gene = split(gene, int)

            data_set = get_dataset_from_json(task_configs[0], fake=True)
            data_set_name = data_set.__class__.__name__

        # find loss and acc in the tensorboard files
        tb_files = find_tb_files(path)
        assert len(tb_files) > 0
        events = read_event_files(tb_files)

        loss_train = events.get("train/loss", None)
        loss_test = events.get("test/loss", None)
        assert (loss_train is not None) and (loss_test is not None)
        accuracy_train = events.get("train/accuracy/1", None)
        accuracy_test = events.get("test/accuracy/1", None)
        assert (accuracy_train is not None) and (accuracy_test is not None)

        # figure out params and flops by building the network
        net_config_path = Builder.find_net_config_path(path)
        network = get_network(net_config_path, data_set.get_data_shape(), data_set.get_label_shape())

        # figure out latency at some point
        pass

        # return result
        average_last = 5
        return MiniResult(
            arch_index=arch_index,
            arch_str="%s(%s)" % (space_name, ", ".join([str(g) for g in gene])),
            arch_tuple=tuple(gene),
            params={data_set_name: network.get_num_parameters()},
            flops={data_set_name: network.profile_macs()},
            latency={data_set_name: -1},
            loss={data_set_name: {
                'train': np.mean([v.value for v in loss_train[-average_last:]]),
                'test': np.mean([v.value for v in loss_test[-average_last:]]),
            }},
            acc1={data_set_name: {
                'train': np.mean([v.value for v in accuracy_train[-average_last:]]),
                'test': np.mean([v.value for v in accuracy_test[-average_last:]]),
            }},
        )

    @classmethod
    def make_from_dirs(cls, path: str, space_name: str, value_space: ValueSpace):
        """
        creating a mini bench dataset by parsing multiple training processes
        """
        results = []
        merged_results = {}
        arch_to_idx = {}
        tuple_to_str = {}
        tuple_to_idx = {}

        # find all results
        task_configs = find_all_files(path, extension=name_task_config)
        assert len(task_configs) > 0
        for i, cfg_path in enumerate(sorted(task_configs)):
            dir_name = os.path.dirname(cfg_path)
            r = cls.make_from_single_dir(dir_name, space_name, arch_index=i)
            if r is None:
                continue
            assert tuple_to_str.get(r.arch_tuple) is None,\
                "can not yet merge duplicate architecture results: %s, in %s" % (r.arch_tuple, dir_name)
            results.append(r)

        # merge duplicates
        len_before = len(results)
        results = MiniResult.merge_result_list(results)
        print("merging: had %d before, merged down to %d" % (len_before, len(results)))

        # build lookup tables
        for i, r in enumerate(results):
            r.arch_index = i
            merged_results[i] = r
            arch_to_idx[r.arch_str] = i
            tuple_to_idx[r.arch_tuple] = i
            tuple_to_str[r.arch_tuple] = r.arch_str

        # build bench set
        data_sets = list(merged_results.get(0).params.keys())
        return MiniNASParsedTabularBenchmark(
            default_data_set=data_sets[0],
            default_result_type='test',
            bench_name="%s on %s" % (space_name, data_sets[0]),
            bench_description="parsed empirical results",
            value_space=value_space, results=merged_results, arch_to_idx=arch_to_idx,
            tuple_to_str=tuple_to_str, tuple_to_idx=tuple_to_idx)


def create_fairnas(path: str) -> MiniNASParsedTabularBenchmark:
    space_name = "fairnas"
    value_space = ValueSpace(*[DiscreteValues.interval(0, 6) for _ in range(19)])
    return MiniNASParsedTabularBenchmark.make_from_dirs(path, space_name, value_space)


def create_bench201(path: str) -> MiniNASParsedTabularBenchmark:
    space_name = "bench201"
    value_space = ValueSpace(*[DiscreteValues.interval(0, 5) for _ in range(6)])
    return MiniNASParsedTabularBenchmark.make_from_dirs(path, space_name, value_space)


def sample_architectures(mini: MiniNASParsedTabularBenchmark, n=10):
    """ sample some random architectures to train """
    for i in range(n):
        print(i, '\t\t', mini.get_value_space().random_sample())


if __name__ == '__main__':
    Builder()

    # path_ = replace_standard_paths('{path_data}/bench/sin/bench201_n5c16_1.pt')
    # mini_ = create_bench201("{path_cluster}/full/test_s3_bench/Cifar100Data/bench201_n5c16/")

    # path_ = replace_standard_paths('{path_data}/bench/sin/bench201_n4c64_1.pt')
    # mini_ = create_bench201("{path_cluster}/full/test_s3_bench/Cifar100Data/bench201_n4c64/")

    # path_ = replace_standard_paths('{path_data}/bench/sin/SIN_fairnas_small_only1.pt')
    # mini_ = create_fairnas("{path_cluster}/full/s3sin/SubImagenetMV100Data/fairnas/1/")

    path_ = replace_standard_paths('{path_data}/bench/sin/SIN_fairnas_v0.1.pt')
    # mini_ = create_fairnas("{path_cluster}/full/s3sin/SubImagenetMV100Data/fairnas/")

    mini_ = MiniNASParsedTabularBenchmark.load(path_)

    sample_architectures(mini_)

    explore(mini_)
    plot(mini_, ['flops', 'acc1'], [False, True], add_pareto=True)

    """
    # average acc of the top 10 networks
    mini_.set_default_result_type("test")
    accs = []
    for r in mini_.get_all_sorted(["acc1"], maximize=[True]):
        accs.append(r.get_acc1())
    print(sum(accs[:10]) / 10)
    """

    mini_.save(path_)
