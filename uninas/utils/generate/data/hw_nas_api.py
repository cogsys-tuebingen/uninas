"""
creating a data set from HW-NAS bench201 latency data
creating tabular lookup models for fbnet data

https://openreview.net/pdf?id=_0kaDkv3dVf
https://github.com/RICE-EIC/HW-NAS-Bench

the code here is mostly adapted from the original
"""


import os
import random
import pandas as pd
from collections import defaultdict
from uninas.models.other.tabular import TabularSumModel
from uninas.utils.paths import replace_standard_paths
from uninas.builder import Builder


def create_nasbench201(bench_path="./HW-NAS-Bench-v1_0.pickle", save_dir="./hw_nas_api/"):
    from hw_nas_bench_api import HWNASBenchAPI
    from uninas.data.datasets.profiled import ProfiledData
    from uninas.optimization.benchmarks.mini.tabular_nats_bench import MiniNATSBenchTabularBenchmark

    assert os.path.isfile(bench_path), "Is not a file: %s" % bench_path
    hw_api = HWNASBenchAPI(bench_path, search_space="nasbench201")

    data_names = list(hw_api.HW_metrics[hw_api.search_space].keys())
    metrics = list(hw_api.HW_metrics[hw_api.search_space][data_names[0]].keys())
    metrics.remove("config")
    sizes = tuple([5]*6)

    df_data = defaultdict(dict)
    df_data_acc = defaultdict(dict)

    for data_name in data_names:
        architectures = hw_api.HW_metrics[hw_api.search_space][data_name]["config"]
        arcs_as_idx = [MiniNATSBenchTabularBenchmark.get_arc_tuples_from_str(v['arch_str'])[1] for v in architectures]

        for metric in metrics:
            metric_values = hw_api.HW_metrics[hw_api.search_space][data_name][metric]
            data = {k: v for k, v in zip(arcs_as_idx, metric_values)}
            assert len(data) == len(arcs_as_idx), "Redundant architectures?"

            # save pytorch data set
            path_save = "%s/%s-%s.pt" % (save_dir, data_name, metric)
            ProfiledData.separate_and_save(path_save, data=data, sizes=sizes, num_test=2000, shuffle=True)

            # add to csv
            key = "%s-%s" % (data_name, metric)
            for k, v in data.items():
                df_data[str(k)][key] = v

            print("saved %s" % path_save)

    # save dataframes
    df = pd.DataFrame(df_data)
    path_save = "%s/hwnas.csv" % save_dir
    # df.to_csv(path_save)
    print("saved full csv: %s" % path_save)


def create_fbnet(bench_path="./HW-NAS-Bench-v1_0.pickle", save_dir="./hw_nas_api/"):
    """
    create tabular lookup models from the bench data
    :param bench_path:
    :param save_dir:
    :return:
    """
    from hw_nas_bench_api import HWNASBenchAPI

    assert os.path.isfile(bench_path_), "Is not a file: %s" % bench_path_
    hw_api = HWNASBenchAPI(bench_path, search_space="fbnet")

    data_sets = ["cifar100", "ImageNet"]
    lookup_tables = hw_api.get_op_lookup_tables()

    stem_channel = 16
    num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    conv_block_list = [
        {"exp": 1, "kernel": 3, "group": 1},
        {"exp": 1, "kernel": 3, "group": 2},
        {"exp": 3, "kernel": 3, "group": 1},
        {"exp": 6, "kernel": 3, "group": 1},
        {"exp": 1, "kernel": 5, "group": 1},
        {"exp": 1, "kernel": 5, "group": 2},
        {"exp": 3, "kernel": 5, "group": 1},
        {"exp": 6, "kernel": 5, "group": 1}
    ]
    num_ops = len(conv_block_list) + 1

    for dataname in data_sets:
        for name, lookup_table in lookup_tables.items():
            if dataname == "cifar100":
                num_classes = 100
                stride_list = [1, 1, 2, 2, 1, 2, 1]
                header_channel = 1504  # FBNetv1 setting
                stride_init = 1
                H_W = 32
            elif dataname == "ImageNet":
                num_classes = 1000  # ImageNet
                stride_list = [1, 2, 2, 2, 1, 2, 1]  # FBNetv1 setting, offcial ImageNet setting
                header_channel = 1984  # FBNetv1 setting
                stride_init = 2
                H_W = 224
            else:
                raise NotImplementedError

            dynamic = defaultdict(list)
            constant = 0.0

            # Add STEM cost
            constant += lookup_table["ConvNorm_H{}_W{}_Cin{}_Cout{}_kernel{}_stride{}_group{}".format(
                H_W,
                H_W,
                3,
                stem_channel,
                3,
                stride_init,
                1,
            )]

            H_W = H_W // stride_init  # Downsample size due to stride
            # Add Cells cost
            layer_id = 0
            for stage_id, num_layer in enumerate(num_layer_list):
                for i in range(num_layer):
                    if i == 0:
                        cur_channels = stem_channel if stage_id == 0 else num_channel_list[stage_id - 1]
                        cur_stride = stride_list[stage_id]
                    else:
                        cur_channels = num_channel_list[stage_id]
                        cur_stride = 1

                    for layer_op_idx in range(num_ops):
                        if layer_op_idx < 8:
                            # ConvBlock
                            lookup_key = "ConvBlock_H{}_W{}_Cin{}_Cout{}_exp{}_kernel{}_stride{}_group{}".format(
                                H_W,
                                H_W,
                                cur_channels,
                                num_channel_list[stage_id],
                                conv_block_list[layer_op_idx]["exp"],
                                conv_block_list[layer_op_idx]["kernel"],
                                cur_stride,
                                conv_block_list[layer_op_idx]["group"],
                            )
                        else:
                            # Skip connection
                            lookup_key = "Skip_H{}_W{}_Cin{}_Cout{}_stride{}".format(
                                H_W,
                                H_W,
                                cur_channels,
                                num_channel_list[stage_id],
                                cur_stride
                            )
                        dynamic[layer_id].append(lookup_table[lookup_key])
                    if i == 0:
                        # first Conv takes the stride into consideration
                        H_W = H_W // stride_list[stage_id]  # Downsample size due to stride
                    layer_id += 1

            # Add Header cost
            constant += lookup_table["ConvNorm_H{}_W{}_Cin{}_Cout{}_kernel{}_stride{}_group{}".format(
                H_W,
                H_W,
                num_channel_list[-1],
                header_channel,
                1,
                1,
                1,
            )]
            # Add AvgP cost
            constant += lookup_table["AvgP_H{}_W{}_Cin{}_Cout{}_kernel{}_stride{}".format(
                H_W,
                H_W,
                header_channel,
                header_channel,
                H_W,
                1,
            )]
            # Add FC cost
            constant += lookup_table["FC_Cin{}_Cout{}".format(
                header_channel,
                num_classes,
            )]

            lookup_model = TabularSumModel(table=dynamic, constant=constant)
            path = "%s/tab-fbnet-%s-%s.pt" % (save_dir, dataname, name)
            lookup_model.save(path=path)
            print("saved %s" % path)


if __name__ == '__main__':
    Builder()
    random.seed(0)
    bench_path_ = replace_standard_paths("{path_data}/bench/HW-NAS-Bench-v1_0.pickle")
    create_nasbench201(bench_path=bench_path_, save_dir=replace_standard_paths("{path_data}/profiling/HW-NAS/"))
    # create_fbnet(bench_path=bench_path_, save_dir=replace_standard_paths("{path_profiled}/HW-NAS/"))
