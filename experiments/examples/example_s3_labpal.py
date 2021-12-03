from uninas.main import Main

"""
retraining a network from a network config (only referenced via name, e.g. FairNasC)

beware that we may be using fake data
"""


args = {
    "cls_task": "SingleRetrainTask",
    "{cls_task}.save_dir": "{path_tmp}/s3/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    "cls_device": "CudaDevicesManager",   # CpuDevicesManager, CudaDevicesManager, TestCpuDevicesManager
    "{cls_device}.num_devices": 1,

    "cls_trainer": "SimpleTrainer",  # SimpleTrainer, SimpleDDPTrainer
    "{cls_trainer}.max_epochs": 5,

    # "cls_clones": "EMAClone",
    # "{cls_clones#0}.device": "cpu",
    # "{cls_clones#0}.decay": 0.99,

    "cls_exp_loggers": "TensorBoardExpLogger",
    "{cls_exp_loggers#0}.log_graph": False,

    "cls_callbacks": "CheckpointCallback",
    "{cls_callbacks#0}.top_n": 1,
    "{cls_callbacks#0}.key": "train/loss",
    "{cls_callbacks#0}.minimize_key": True,

    "cls_method": "RetrainMethod",

    "cls_network": "RetrainUninasNetwork",
    "{cls_network}.config_path": "FairNasC",

    "cls_data": "Imagenet1000Data",
    "{cls_data}.batch_size_train": 16,
    "{cls_data}.fake": False,
    "{cls_data}.dir": '{path_data}/ImageNet_ILSVRC2012/',
    "{cls_data}.valid_as_test": True,

    "cls_augmentations": "TimmImagenetAug",
    "{cls_augmentations#0}.crop_size": 224,

    # "cls_data": "Cifar100Data",
    # "{cls_data}.fake": True,
    # "{cls_data}.batch_size_train": 4,
    # "{cls_data}.num_workers": 0,

    # "cls_augmentations": "DartsCifarAug, CutoutAug",
    # "{cls_augmentations#1}.size": 16,

    "cls_metrics": "AccuracyMetric",

    "cls_initializers": "",

    "cls_criterion": "CrossEntropyCriterion",
    "{cls_criterion}.smoothing_epsilon": 0.1,

    "cls_optimizers": "LabPalOptimizer",
    "{cls_optimizers#0}.mode": "SGD",
    "{cls_optimizers#0}.noise_factor": 25.0,
    "{cls_optimizers#0}.update_step_adaptation": 1.0,
    "{cls_optimizers#0}.momentum": 0.0,
    "{cls_optimizers#0}.amount_steps_to_reuse_lr": 1000,
    "{cls_optimizers#0}.amount_batches_for_full_batch_loss_approximation": 10,
    "{cls_optimizers#0}.batch_size_schedule_steps": "0, 80, 260",
    "{cls_optimizers#0}.batch_size_schedule_factor": "1, 2, 4",
    "{cls_optimizers#0}.amount_untrustworthy_initial_steps": 400,
    "{cls_optimizers#0}.amount_of_former_lr_values_to_consider": 15,
    "{cls_optimizers#0}.parabolic_approximation_sample_step_size": 0.01,
    "{cls_optimizers#0}.max_step_size": 1.0,
    "{cls_optimizers#0}.epsilon": 1e-10,
    "{cls_optimizers#0}.is_print": True,

    "cls_regularizers": "DropOutRegularizer",
    "{cls_regularizers#0}.prob": 0.2,
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    # print(task.get_method().get_network())
    task.load()
    # task.load('{path_tmp}/s3_2/')
    task.run()
