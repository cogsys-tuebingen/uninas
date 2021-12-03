from uninas.main import Main

"""
training a super net

beware that we are using fake data and a test run (only 10 steps/epoch)
"""

config_files = "{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/fairnas.run_config"


args = {
    "cls_task": "SingleSearchTask",
    "{cls_task}.save_dir": "{path_tmp}/s1_mini/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,
    "{cls_task}.is_deterministic": False,

    "cls_device": "CudaDevicesManager",  # CpuDevicesManager, CudaDevicesManager, TestCpuDevicesManager
    "{cls_device}.num_devices": 1,

    "cls_trainer": "SimpleTrainer",
    "{cls_trainer}.max_epochs": 10,
    "{cls_trainer}.eval_last": 2,
    "{cls_trainer}.test_last": 2,

    # "cls_clones": "EMAClone, EMAClone",
    # "{cls_clones#0}.device": "same",
    # "{cls_clones#0}.decay": 0.99,
    # "{cls_clones#1}.device": "same",
    # "{cls_clones#1}.decay": 0.999,

    "cls_exp_loggers": "TensorBoardExpLogger",

    "cls_callbacks": "CheckpointCallback",
    # "cls_callbacks": "CheckpointCallback, SplitWeightsMixedOpCallback",
    "{cls_callbacks#0}.save_clone": True,
    "{cls_callbacks#0}.top_n": 0,
    "{cls_callbacks#0}.key": "train/loss",
    "{cls_callbacks#0}.minimize_key": True,
    # "{cls_callbacks#1}.milestones": "5",

    "cls_data": "Imagenet1000Data",
    "{cls_data}.fake": True,
    "{cls_data}.batch_size_train": 2,
    "{cls_data}.valid_split": 12800,
    "{cls_data}.valid_as_test": True,

    "cls_augmentations": "TimmImagenetAug",
    "{cls_augmentations#0}.crop_size": 224,

    "cls_method": "UniformRandomMethod",  # StrictlyFairRandomMethod, UniformRandomMethod
    "{cls_method}.mask_indices": "",

    "cls_network": "SearchUninasNetwork",

    "cls_network_body": "StackedCellsNetworkBody",
    "{cls_network_body}.cell_order": "s0-n, s0-n, s1-r, s1-n, s1-n",

    "cls_network_cells": "SingleLayerCNNSearchCell, SingleLayerCNNSearchCell, SingleLayerCNNSearchCell",
    "{cls_network_cells#0}.name": "s0-n",
    "{cls_network_cells#0}.arc_key": "s0-n",
    "{cls_network_cells#0}.arc_shared": True,
    "{cls_network_cells#0}.features_fixed": 16,
    "{cls_network_cells#0}.stride": 2,
    "{cls_network_cells#1}.name": "s1-r",
    "{cls_network_cells#1}.arc_key": "s1-r",
    "{cls_network_cells#1}.arc_shared": True,
    "{cls_network_cells#1}.features_fixed": 32,
    "{cls_network_cells#1}.stride": 2,
    "{cls_network_cells#2}.name": "s1-n",
    "{cls_network_cells#2}.arc_key": "s1-n",
    "{cls_network_cells#2}.arc_shared": False,
    "{cls_network_cells#2}.features_fixed": 32,
    "{cls_network_cells#2}.stride": 1,

    "cls_network_cells_primitives": "MobileNetV2SkipLTPrimitives, MobileNetV2SkipLTPrimitives, MobileNetV2SkipLTPrimitives",
    "{cls_network_cells_primitives#0}.subset": "3, 4, 5",
    "{cls_network_cells_primitives#0}.mixed_cls": "MixedOp",  # MixedOp, BiasMixedOp, AttentionSigmoidMixedOp, SplitWeightsMixedOp
    "{cls_network_cells_primitives#1}.mixed_cls": "MixedOp",
    "{cls_network_cells_primitives#2}.mixed_cls": "MixedOp",

    "cls_network_stem": "MobileNetV2Stem",
    "{cls_network_stem}.features": 16,
    "{cls_network_stem}.stride": 2,
    "{cls_network_stem}.k_size": 3,
    "{cls_network_stem}.act_fun": "swish",
    "{cls_network_stem}.k_size1": 3,
    "{cls_network_stem}.act_fun1": "swish",

    "cls_network_heads": "FeatureMixClassificationHead",
    "{cls_network_heads#0}.weight": 1.0,
    "{cls_network_heads#0}.cell_idx": -1,
    "{cls_network_heads#0}.persist": "True",
    "{cls_network_heads#0}.features": 128,
    "{cls_network_heads#0}.act_fun": "relu6",

    "cls_metrics": "AccuracyMetric",

    "cls_initializers": "SPOSInitializer",

    "cls_regularizers": "DropPathRegularizer",
    "{cls_regularizers#0}.min_prob": 0.2,
    "{cls_regularizers#0}.max_prob": 0.3,

    "cls_criterion": "CrossEntropyCriterion",

    "cls_optimizers": "SGDOptimizer",
    "{cls_optimizers#0}.lr": 0.05,
    "{cls_optimizers#0}.momentum": 0.9,
    "{cls_optimizers#0}.clip_abs_value": 5.0,

    "cls_schedulers": "ExponentialScheduler",
    "{cls_schedulers#0}.gamma": 0.9,
    "{cls_schedulers#0}.each_samples": 3000000,
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    # print(task.get_method().get_network())
    task.load()
    task.run()
