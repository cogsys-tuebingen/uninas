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

    "cls_trainer": "SimpleTrainer",  # SimpleTrainer, SimpleDDPTrainer, LightningTrainer
    "{cls_trainer}.max_epochs": 5,

    "cls_clones": "EMAClone",
    "{cls_clones#0}.device": "cpu",
    "{cls_clones#0}.decay": 0.99,

    "cls_exp_loggers": "TensorBoardExpLogger",
    "{cls_exp_loggers#0}.log_graph": False,

    "cls_callbacks": "CheckpointCallback",
    "{cls_callbacks#0}.top_n": 1,
    "{cls_callbacks#0}.key": "train/loss",
    "{cls_callbacks#0}.minimize_key": True,

    "cls_method": "RetrainMethod",

    # "cls_network": "EfficientNetTimmNetwork",
    # "{cls_network}.model_name": "tf_efficientnet_b0",
    # "{cls_network}.checkpoint_path": "{path_pretrained}",

    "cls_network": "RetrainUninasNetwork",
    "{cls_network}.config_path": "FairNasC",
    # "{cls_network}.config_path": "ShuffleNetV2PlusMedium",
    # "{cls_network}.config_path": "EfficientNetB0",
    # "{cls_network}.config_path": "imagenet_small",
    # maybe amp problematic: ScarletNasC, SPOSNet, ShuffleNetV2PlusMedium, ResNet18
    # "{cls_network}.checkpoint_path": "{path_pretrained}",

    # "cls_network": "RetrainFromSearchUninasNetwork",  # build from a search space
    # "{cls_network}.search_config_path": "{path_tmp}/run_config/",
    # "{cls_network}.gene": "random",

    "cls_data": "Imagenet1000Data",
    "{cls_data}.batch_size_train": 4,
    "{cls_data}.fake": False,
    "{cls_data}.dir": '{path_data}/ImageNet_ILSVRC2012/',
    "{cls_data}.valid_as_test": True,

    # "cls_data": "SubImagenet100Data",  # disable pretrained weights, otherwise there will be a weight mismatch
    # "{cls_data}.dir": "{path_data}/subImageNet/",
    # "{cls_data}.fake": False,
    # "{cls_data}.valid_split": 0.1,
    # "{cls_data}.batch_size_train": 2,

    # "cls_augmentations": "DartsImagenetAug",
    "cls_augmentations": "TimmImagenetAug, MixUpAug",
    "{cls_augmentations#0}.crop_size": 224,
    "{cls_augmentations#1}.alpha1": 1.0,
    "{cls_augmentations#1}.alpha2": 1.0,

    "cls_metrics": "AccuracyMetric",

    "cls_initializers": "",
    # "cls_initializers": "LoadWeightsInitializer",
    # "{cls_initializers#0}.path": "{path_tmp}/run_config/",
    # "{cls_initializers#0}.gene": "1, 0,   2, 0, 0, 0,   0, 0, 0, 3,   0, 0, 0, 0,   5, 5, 3, 3,   4",
    # "{cls_initializers#0}.strict": True,

    "cls_criterion": "CrossEntropyCriterion",

    "cls_optimizers": "SGDOptimizer",
    "{cls_optimizers#0}.lr": 0.00,
    "{cls_optimizers#0}.momentum": 0.9,

    "cls_schedulers": "CosineScheduler",
    "{cls_schedulers#0}.warmup_epochs": 2,
    "{cls_schedulers#0}.warmup_lr": 0.01,

    "cls_regularizers": "DropOutRegularizer",
    # "cls_regularizers": "DropOutRegularizer, DropPathRegularizer",
    "{cls_regularizers#0}.prob": 0.0,
    # "{cls_regularizers#1}.max_prob": 0.1,

    # "{cls_method}.amp_enabled": False,
    # "{cls_optimizers#0}.weight_decay": 4e-3,
    # "{cls_optimizers#0}.weight_decay_filter": True,
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    # print(task.get_method().get_network())
    task.load()
    # task.load('{path_tmp}/s3_2/')
    task.run()
