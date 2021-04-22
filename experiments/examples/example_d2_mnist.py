from uninas.main import Main

"""
retraining a network from a network config, here is some DARTS-space specific stuff (e.g. drop path)
"""


args = {
    "cls_task": "'SingleRetrainTask'",
    "{cls_task}.save_dir": "{path_tmp}/d2/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    "cls_device": "CudaDevicesManager",
    "{cls_device}.num_devices": 1,

    "cls_trainer": "SimpleTrainer",
    "{cls_trainer}.max_epochs": 10,
    "{cls_trainer}.eval_last": 3,
    "{cls_trainer}.test_last": 3,

    "cls_clones": "EMAClone",
    "{cls_clones#0}.device": "same",
    "{cls_clones#0}.decay": 0.9,

    "cls_exp_loggers": "TensorBoardExpLogger",
    "{cls_exp_loggers#0}.log_graph": False,

    "cls_callbacks": "CheckpointCallback",
    "{cls_callbacks#0}.save_clone": True,
    "{cls_callbacks#0}.top_n": 0,
    "{cls_callbacks#0}.key": "train/loss",
    "{cls_callbacks#0}.minimize_key": True,

    "cls_data": "MnistData",
    "{cls_data}.fake": False,
    "{cls_data}.valid_split": 0.2,
    "{cls_data}.dir": "{path_data}/mnist/",
    "{cls_data}.download": True,
    "{cls_data}.batch_size_train": 256,

    "cls_augmentations": "DartsCifarAug",

    "cls_method": "RetrainMethod",

    "cls_network": "RetrainInsertConfigUninasNetwork",
    "{cls_network}.config_path": "DARTS_V1",
    # "{cls_network}.config_path": "/{path_tmp}/from_config/finalized.network_config",

    "{cls_network_body}.cell_order": "n, r, n, r, n",

    "{cls_network_stem}.features": 8,

    "cls_network_heads": "ClassificationHead",
    "{cls_network_heads#0}.weight": 1.0,
    "{cls_network_heads#0}.cell_idx": -1,
    "{cls_network_heads#0}.persist": "True",

    "cls_metrics": "AccuracyMetric, ConfusionMatrixMetric, CriterionMetric, CriterionMetric",
    "{cls_metrics#1}.each_epochs": 2,
    "{cls_metrics#2}.criterion": 'L1Criterion',
    "{cls_metrics#3}.criterion": 'L2Criterion',

    "cls_initializers": "",

    "cls_regularizers": "DropOutRegularizer",
    "{cls_regularizers#0}.prob": 0.5,

    "cls_criterion": "CrossEntropyCriterion",

    "cls_optimizers": "RMSPropOptimizer",
    "{cls_optimizers#0}.lr": 0.008,
    "{cls_optimizers#0}.alpha": 0.9,
    "{cls_optimizers#0}.weight_decay": 1e-5,

    "cls_schedulers": "CosineScheduler",
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
