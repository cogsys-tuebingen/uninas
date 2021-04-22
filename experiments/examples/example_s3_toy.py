from uninas.main import Main

"""
training a fully connected network on toy data
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
    "{cls_trainer}.accumulate_batches": 1,

    "cls_clones": "EMAClone",
    "{cls_clones#0}.device": "same",
    "{cls_clones#0}.decay": 0.9,

    "cls_exp_loggers": "TensorBoardExpLogger",

    "cls_callbacks": "CheckpointCallback",
    "{cls_callbacks#0}.top_n": 1,
    "{cls_callbacks#0}.key": "train/loss",
    "{cls_callbacks#0}.minimize_key": True,

    "cls_method": "RetrainMethod",

    "cls_network": "FullyConnectedNetwork",
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

    "cls_metrics": "CriterionMetric, CriterionMetric",
    "{cls_metrics#0}.criterion": 'L1Criterion',
    "{cls_metrics#1}.criterion": 'L2Criterion',

    "cls_initializers": "",

    "cls_criterion": "L1Criterion",

    "cls_optimizers": "SGDOptimizer",
    "{cls_optimizers#0}.lr": 0.01,
    "{cls_optimizers#0}.momentum": 0.9,

    "cls_schedulers": "ConstantScheduler",
    # "cls_schedulers": "CosineScheduler",
    # "{cls_schedulers#0}.warmup_epochs": 1,
    # "{cls_schedulers#0}.warmup_lr": 0.0,

    "cls_regularizers": "",
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    # print(task.get_method().get_network())
    task.load()
    # task.load('{path_tmp}/s3_2/')
    task.run()
