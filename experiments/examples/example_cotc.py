from uninas.main import Main

"""
training a super net

beware that we are using fake data and a test run (only 10 steps/epoch)
"""


# search network
config_files = "{path_conf_net_search}/fairnas.run_config"


args = {
    "cls_task": "SingleSearchTask",
    "{cls_task}.save_dir": "{path_tmp}/cotc/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    "cls_device": "CudaDevicesManager",  # CpuDevicesManager, CudaDevicesManager, TestCpuDevicesManager
    "{cls_device}.num_devices": 1,

    "cls_trainer": "SimpleTrainer",
    "{cls_trainer}.max_epochs": 5,
    "{cls_trainer}.eval_last": 0,
    "{cls_trainer}.test_last": 0,

    "cls_exp_loggers": "TensorBoardExpLogger",

    "cls_callbacks": "CheckpointCallback",
    "{cls_callbacks#0}.save_clone": True,
    "{cls_callbacks#0}.top_n": 0,
    "{cls_callbacks#0}.key": "train/loss",
    "{cls_callbacks#0}.minimize_key": True,

    "cls_data": "Imagenet1000Data",
    "{cls_data}.fake": False,
    "{cls_data}.batch_size_train": 16,
    "{cls_data}.dir": "{path_data}/ImageNet_ILSVRC2012/",
    "{cls_data}.valid_split": 12800,
    "{cls_data}.valid_as_test": True,

    "cls_augmentations": "TimmImagenetAug",
    "{cls_augmentations#0}.crop_size": 224,

    "cls_method": "CreamOfTheCropMethod",
    "{cls_method}.mask_indices": "",
    "{cls_method}.board_size": 3,  # 10
    "{cls_method}.pre_prob": "0.05, 0.2, 0.05, 0.5, 0.05, 0.15",
    "{cls_method}.grace_epochs": 0,  # 20
    "{cls_method}.select_teacher": "meta",  # meta, value1, l1, random
    "{cls_method}.select_update_iter": 1,  # 200
    "{cls_method}.match_batch_size": 4,
    "{cls_method}.mmn_batch_average": False,
    "{cls_method}.mmn_batch_size": 8,

    # we can use an actual network as MMN net
    "cls_mmn_network": "FullyConnectedNetwork",
    "{cls_mmn_network}.layer_widths": "20",
    "{cls_mmn_network}.act_fun": "sigmoid",
    "{cls_mmn_network}.use_bn": False,
    "{cls_mmn_network}.use_bias": True,

    "cls_cotc_targets": "OptimizationTarget, OptimizationTarget",
    "{cls_cotc_targets#0}.key": "train/accuracy/1",
    "{cls_cotc_targets#0}.maximize": True,
    "{cls_cotc_targets#1}.key": "train/estimators/macs",
    "{cls_cotc_targets#1}.maximize": False,

    "cls_metrics": "AccuracyMetric",

    "cls_arc_estimators": "ModelEstimator, ModelEstimator",
    # profiled macs, used as optimization objective for the prioritized board
    "{cls_arc_estimators#0}.key": "macs",
    "{cls_arc_estimators#0}.is_objective": True,
    "{cls_arc_estimators#0}.maximize": False,
    "{cls_arc_estimators#0}.model_file_path": "{path_profiled}/tab_fairnas_macs.pt",
    # profiled latency, constraint, prevent slow architectures from being listed in the prioritized board
    "{cls_arc_estimators#1}.key": "latency",
    "{cls_arc_estimators#1}.is_objective": True,
    "{cls_arc_estimators#1}.maximize": False,
    "{cls_arc_estimators#1}.model_file_path": "{path_profiled}/tab_fairnas_latency.pt",
    "{cls_arc_estimators#1}.is_constraint": True,
    "{cls_arc_estimators#1}.min_value": 0.0,
    "{cls_arc_estimators#1}.max_value": 0.01,

    "cls_initializers": "SPOSInitializer",

    "cls_regularizers": "",

    "cls_criterion": "CrossEntropyCriterion",

    "cls_optimizers": "SGDOptimizer",
    "{cls_optimizers#0}.lr": 0.05,
    "{cls_optimizers#0}.momentum": 0.9,
    "{cls_optimizers#0}.weight_decay": 1e-4,

    "cls_schedulers": "CosineScheduler",
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task(config_files, args_changes=args)
    # print(task.get_method().get_network())
    task.load()
    task.run()
