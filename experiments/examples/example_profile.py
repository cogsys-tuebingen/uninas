from uninas.main import Main

"""
profiling a super net
"""


args = {
    # the task is to profile
    "cls_task": "SearchNetworkProfileTask",
    "{cls_task}.save_dir": "{path_tmp}/profile/",
    "{cls_task}.save_del_old": False,
    "{cls_task}.is_test_run": False,

    # device on which the profiling function operates
    "cls_device": "CudaDevicesManager",   # CpuDevicesManager, CudaDevicesManager
    "{cls_device}.num_devices": 1,

    # profiler to select how the profiling and prediction works
    # "cls_profiler": "TabularCellsProfiler",
    "cls_profiler": "SampleArchitecturesProfiler",
    "{cls_profiler}.sample_overcomplete": True,
    "{cls_profiler}.sample_standalone": False,
    "{cls_profiler}.num_train": 20,
    "{cls_profiler}.num_test": 10,

    # function to profile one specific architecture in the super-network
    "cls_profile_fun": "LatencyTimeProfileFunction",  # ParamsProfileFunction
    "{cls_profile_fun}.num_warmup": 5,
    "{cls_profile_fun}.num_measure": 10,

    # data, just to know input/output sizes for the network
    "cls_data": "Imagenet1000Data",
    "{cls_data}.fake": True,
    "{cls_data}.batch_size_train": 16,
    "{cls_data}.valid_as_test": True,

    "cls_augmentations": "DartsImagenetAug",
    "{cls_augmentations#0}.crop_size": 224,

    # network definition
    "cls_network": "SearchUninasNetwork",

    "cls_network_body": "StackedCellsNetworkBody",
    "{cls_network_body}.cell_order": "s0-n, s0-n, s1-r, s1-n, s1-n",

    "cls_network_cells": "SingleLayerCNNSearchCell, SingleLayerCNNSearchCell, SingleLayerCNNSearchCell",
    "{cls_network_cells#0}.name": "s0-n",
    "{cls_network_cells#0}.arc_key": "s0-n",
    "{cls_network_cells#0}.arc_shared": False,
    "{cls_network_cells#0}.features_fixed": 16,
    "{cls_network_cells#0}.stride": 2,
    "{cls_network_cells#1}.name": "s1-r",
    "{cls_network_cells#1}.arc_key": "s1-r",
    "{cls_network_cells#1}.arc_shared": False,
    "{cls_network_cells#1}.features_fixed": 32,
    "{cls_network_cells#1}.stride": 2,
    "{cls_network_cells#2}.name": "s1-n",
    "{cls_network_cells#2}.arc_key": "s1-n",
    "{cls_network_cells#2}.arc_shared": False,
    "{cls_network_cells#2}.features_fixed": 32,
    "{cls_network_cells#2}.stride": 1,

    "cls_network_cells_primitives": "MobileNetV2Primitives, MobileNetV2Primitives, MobileNetV2Primitives",

    "cls_network_stem": "MobileNetV2Stem",
    "{cls_network_stem}.features": 16,
    "{cls_network_stem}.stride": 2,
    "{cls_network_stem}.k_size": 3,
    "{cls_network_stem}.act_fun": "swish",
    "{cls_network_stem}.k_size1": 3,
    "{cls_network_stem}.act_fun1": "hswish",

    "cls_network_heads": "FeatureMixClassificationHead",
    "{cls_network_heads#0}.weight": 1.0,
    "{cls_network_heads#0}.cell_idx": -1,
    "{cls_network_heads#0}.persist": "True",
    "{cls_network_heads#0}.features": 128,
    "{cls_network_heads#0}.act_fun": "relu6",
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
