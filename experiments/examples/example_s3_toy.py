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
    "{cls_trainer}.max_epochs": 20,

    "{cls_trainer}.ema_device": "same",
    "{cls_trainer}.ema_decay": 0.99,

    "cls_exp_loggers": "TensorBoardExpLogger",

    "cls_callbacks": "CheckpointCallback",
    "{cls_callbacks#0}.top_n": 1,
    "{cls_callbacks#0}.key": "train/loss",
    "{cls_callbacks#0}.minimize_key": True,

    "cls_method": "RetrainMethod",

    "cls_network": "FCProfilingNetwork",
    # "{cls_network}.checkpoint_path": "",
    "{cls_network}.layer_widths": "20, 21",
    "{cls_network}.act_fun": "sigmoid",
    "{cls_network}.use_bn": True,
    "{cls_network}.use_bias": True,

    "cls_data": "SumToyData",
    "{cls_data}.valid_split": 0.04,
    "{cls_data}.fake": False,
    "{cls_data}.batch_size_train": 100,
    "{cls_data}.vector_size": 16,

    "cls_augmentations": "",

    "cls_metrics": "",

    "cls_initializers": "",

    "cls_criterion": "L1Criterion",

    "cls_optimizers": "SGDOptimizer",
    "{cls_optimizers#0}.lr": 0.01,
    "{cls_optimizers#0}.momentum": 0.9,
    "{cls_optimizers#0}.accumulate_batches": 2,

    "cls_schedulers": "CosineScheduler",
    "{cls_schedulers#0}.warmup_epochs": 3,
    "{cls_schedulers#0}.warmup_lr": 0.0,

    "cls_regularizers": "",
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    # print(task.get_method().get_network())
    task.load()
    # task.load('{path_tmp}/s3_2/')
    task.run()
