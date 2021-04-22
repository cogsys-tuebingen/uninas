from uninas.main import Main

"""
retraining a network from a network config, here is some DARTS-space specific stuff (e.g. drop path)

beware that we are using fake data
"""


args = {
    "cls_task": "'SingleRetrainTask'",
    "{cls_task}.save_dir": "{path_tmp}/d2/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    "cls_device": "CudaDevicesManager",
    "{cls_device}.num_devices": 1,

    "cls_trainer": "SimpleTrainer",   # SimpleTrainer, LightningTrainer
    "{cls_trainer}.max_epochs": 4,
    "{cls_trainer}.eval_last": 2,
    "{cls_trainer}.test_last": 2,

    "cls_clones": "EMAClone",
    "{cls_clones#0}.device": "same",
    "{cls_clones#0}.decay": 0.9,

    "cls_exp_loggers": "TensorBoardExpLogger",
    "{cls_exp_loggers#0}.log_graph": False,

    "cls_method": "RetrainMethod",

    "cls_network": "RetrainInsertConfigUninasNetwork",
    "{cls_network}.config_path": "DARTS_V1",
    # "{cls_network}.config_path": "/{path_tmp}/from_config/finalized.network_config",

    "cls_data": "Cifar100Data",
    "{cls_data}.fake": True,
    "{cls_data}.batch_size_train": 4,

    "cls_augmentations": "DartsCifarAug, CutoutAug",
    "{cls_augmentations#1}.size": 16,

    # if not specified, use the same cls_network_body as in the network config
    "{cls_network_body}.cell_order": "n, n, n, r, n, n, n, n, r, n, n",
    "{cls_network_body}.features_first_cell": 64,  # usually stem features * 3 on cifar

    "{cls_network_stem}.features": 16*4,  # usually 36*4

    "cls_network_heads": "ClassificationHead",
    "{cls_network_heads#0}.weight": 1.0,
    "{cls_network_heads#0}.cell_idx": -1,
    "{cls_network_heads#0}.persist": "True",

    "cls_metrics": "AccuracyMetric",

    "cls_initializers": "",

    "cls_regularizers": "DropPathRegularizer",
    "{cls_regularizers#0}.max_prob": 0.3,

    "cls_criterion": "CrossEntropyCriterion",

    "cls_optimizers": "SGDOptimizer",
    "{cls_optimizers#0}.lr": 0.01,
    "{cls_optimizers#0}.momentum": 0.7,

    "cls_schedulers": "CosineScheduler",
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
