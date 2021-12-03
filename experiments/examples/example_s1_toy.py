from uninas.main import Main

"""
training a super net
"""


args = {
    "cls_task": "SingleSearchTask",
    "{cls_task}.save_dir": "{path_tmp}/s1_toy/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,
    "{cls_task}.is_deterministic": False,

    "cls_device": "CudaDevicesManager",  # CpuDevicesManager, CudaDevicesManager, TestCpuDevicesManager
    "{cls_device}.num_devices": 1,

    "cls_trainer": "SimpleTrainer",
    "{cls_trainer}.max_epochs": 10,
    "{cls_trainer}.eval_last": 2,
    "{cls_trainer}.test_last": 2,
    # "{cls_trainer}.sharing_strategy": 'file_system',

    # "cls_clones": "EMAClone",
    # "{cls_clones#0}.device": "same",
    # "{cls_clones#0}.decay": 0.9,

    "cls_exp_loggers": "TensorBoardExpLogger",

    "cls_callbacks": "CheckpointCallback",
    "{cls_callbacks#0}.save_clone": True,
    "{cls_callbacks#0}.top_n": 0,
    "{cls_callbacks#0}.key": "train/loss",
    "{cls_callbacks#0}.minimize_key": True,

    "cls_data": "SumToyData",
    "{cls_data}.fake": False,
    "{cls_data}.batch_size_train": 100,

    # "cls_data": "RandomRegressionData",
    # "{cls_data}.batch_size_train": 200,
    # "{cls_data}.data_shape": "20",
    # "{cls_data}.target_shape": "1",

    "cls_augmentations": "",

    "cls_method": "DSNASMethod",  # StrictlyFairRandomMethod, UniformRandomMethod, DSNASMethod
    "{cls_method}.mask_indices": "0, 2, 4",

    "cls_network": "SearchUninasNetwork",

    "cls_network_body": "StackedCellsNetworkBody",
    "{cls_network_body}.cell_order": "s0-n, s0-n, s1-n, s1-n",

    "cls_network_cells": "SingleLayerSearchCell, SingleLayerSearchCell",
    "{cls_network_cells#0}.name": "s0-n",
    "{cls_network_cells#0}.arc_key": "s0-n",
    "{cls_network_cells#0}.arc_shared": True,
    "{cls_network_cells#0}.features_fixed": 48,
    "{cls_network_cells#1}.name": "s1-n",
    "{cls_network_cells#1}.arc_key": "s1-n",
    "{cls_network_cells#1}.arc_shared": False,
    "{cls_network_cells#1}.features_fixed": 64,

    "cls_network_cells_primitives": "MobileNetV2Primitives, MobileNetV2Primitives",

    "cls_network_stem": "LinearToConvStem",
    "{cls_network_stem}.features": 32,
    "{cls_network_stem}.act_fun": "swish",

    "cls_network_heads": "FeatureMixClassificationHead",
    "{cls_network_heads#0}.weight": 1.0,
    "{cls_network_heads#0}.cell_idx": -1,
    "{cls_network_heads#0}.persist": "True",
    "{cls_network_heads#0}.features": 32,
    "{cls_network_heads#0}.act_fun": "relu6",

    "cls_metrics": "",

    "cls_initializers": "",

    "cls_regularizers": "",

    "cls_criterion": "L1Criterion",

    "cls_optimizers": "SGDOptimizer",
    "{cls_optimizers#0}.lr": 0.01,
    "{cls_optimizers#0}.momentum": 0.9,

    "cls_schedulers": "CosineScheduler",
    "{cls_schedulers#0}.warmup_epochs": 2,
    "{cls_schedulers#0}.warmup_lr": 0.001,
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    # print(task.get_method().get_network())
    task.load()
    task.run()
