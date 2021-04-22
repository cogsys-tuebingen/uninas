import time
from uninas.main import Main

"""
Population Based Training (PBT)
- multiple clients like this connect to a server and receive instructions to save/load and change hyperparameters

start this n times in different processes
retraining a network from a network config (only referenced via name, e.g. FairNasC)

beware that we may be using fake data
"""


args = {
    "cls_task": "SingleRetrainTask",
    "{cls_task}.save_dir": "{path_tmp}/pbt_client/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    "cls_device": "CudaDevicesManager",   # CpuDevicesManager, CudaDevicesManager, TestCpuDevicesManager
    "{cls_device}.num_devices": 1,

    "cls_trainer": "SimpleTrainer",  # SimpleTrainer, SimpleDDPTrainer, LightningTrainer
    "{cls_trainer}.max_epochs": 4,
    "{cls_trainer}.eval_last": -1,
    "{cls_trainer}.test_last": -1,

    "cls_clones": "EMAClone",
    "{cls_clones#0}.device": "same",
    "{cls_clones#0}.decay": 0.999,

    "cls_exp_loggers": "TensorBoardExpLogger",
    "{cls_exp_loggers#0}.log_graph": False,

    # this callback is what causes PBT training, the path must match the one in the server
    "cls_callbacks": "PbtCallback",
    "{cls_callbacks#0}.communication_file": "{path_tmp}/pbt_server/communication_uri",

    "cls_method": "RetrainMethod",

    "cls_network": "RetrainUninasNetwork",
    "{cls_network}.config_path": "imagenet_small",

    "cls_data": "Imagenet1000Data",
    "{cls_data}.batch_size_train": 2,
    "{cls_data}.fake": False,
    "{cls_data}.dir": '{path_data}/ImageNet_ILSVRC2012/',
    "{cls_data}.valid_split": 0.1,
    "{cls_data}.valid_as_test": True,

    # "cls_augmentations": "DartsImagenetAug",
    "cls_augmentations": "TimmImagenetAug, MixUpAug",
    "{cls_augmentations#0}.crop_size": 224,
    "{cls_augmentations#1}.alpha1": 1.0,
    "{cls_augmentations#1}.alpha2": 1.0,

    "cls_metrics": "AccuracyMetric",

    "cls_initializers": "",

    "cls_criterion": "CrossEntropyCriterion",

    "cls_optimizers": "SGDOptimizer",
    "{cls_optimizers#0}.lr": 0.005,
    "{cls_optimizers#0}.momentum": 0.9,

    "cls_schedulers": "",

    "cls_regularizers": "DropOutRegularizer",
    # "cls_regularizers": "DropOutRegularizer, DropPathRegularizer",
    "{cls_regularizers#0}.prob": 0.321,
    # "{cls_regularizers#1}.max_prob": 0.1,

    # "{cls_method}.amp_enabled": False,
    # "{cls_optimizers#0}.weight_decay": 4e-3,
    # "{cls_optimizers#0}.weight_decay_filter": True,
}

if __name__ == "__main__":
    t = int(time.time())
    # ignore the command line, use "args" instead
    args["{cls_task}.seed"] = t
    args["{cls_task}.save_dir"] = '%s/%d/' % (args.get("{cls_task}.save_dir"), t)
    task = Main.new_task([], args_changes=args)
    # print(task.get_method().get_network())
    task.load()
    # task.load('{path_tmp}/s3_2/')
    task.run()
