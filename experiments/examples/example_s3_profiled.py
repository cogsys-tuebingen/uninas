from uninas.main import Main

"""
training a fully connected network on profiling data
"""


args = {
    "cls_task": "SingleRetrainTask",
    "{cls_task}.save_dir": "{path_tmp}/s3/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    "cls_device": "CudaDevicesManager",   # CpuDevicesManager, CudaDevicesManager, TestCpuDevicesManager
    "{cls_device}.num_devices": 1,

    "cls_trainer": "SimpleTrainer",  # SimpleTrainer, SimpleDDPTrainer
    "{cls_trainer}.max_epochs": 10,  # 100
    "{cls_trainer}.accumulate_batches": 1,

    "cls_clones": "EMAClone",
    "{cls_clones#0}.device": "cpu",
    "{cls_clones#0}.decay": 0.99,

    "cls_exp_loggers": "TensorBoardExpLogger",

    "cls_callbacks": "CheckpointCallback",
    "{cls_callbacks#0}.top_n": 1,
    "{cls_callbacks#0}.key": "train/loss",
    "{cls_callbacks#0}.minimize_key": True,
    "{cls_callbacks#0}.save_type": "model",  # model, state

    "cls_method": "RetrainMethod",

    "cls_network": "FullyConnectedNetwork",
    "{cls_network}.layer_widths": "144, 20",
    "{cls_network}.act_fun": "sigmoid",
    "{cls_network}.use_bn": False,
    "{cls_network}.use_bias": False,

    "cls_data": "ProfiledData",
    "{cls_data}.valid_split": 0.1,
    "{cls_data}.batch_size_train": 100,
    # "{cls_data}.dir": "{path_data}/profiling/fairnas/cpu/",
    # "{cls_data}.file_name": "data_standalone.pt",
    # "{cls_data}.file_name": "data_overcomplete.pt",
    "{cls_data}.dir": "{path_data}/profiling/HW-NAS/",
    # "{cls_data}.file_name": "cifar10-edgegpu_energy.pt",
    "{cls_data}.file_name": "ImageNet16-120-raspi4_latency.pt",
    "{cls_data}.cast_one_hot": True,
    "{cls_data}.scale": 1.0,
    "{cls_data}.train_num": -1,

    "cls_augmentations": "",

    "cls_metrics": "CriterionMetric, CriterionMetric, CriterionMetric, CriterionMetric, CorrelationsMetric",
    "{cls_metrics#0}.criterion": 'RelativeL1Criterion',
    "{cls_metrics#1}.criterion": 'L1Criterion',
    "{cls_metrics#2}.criterion": 'L2Criterion',
    "{cls_metrics#3}.criterion": 'Huber1Criterion',
    "{cls_metrics#4}.correlations": 'KendallTauNasMetric, SpearmanNasMetric',
    "{cls_metrics#4}.each_epochs": 1,

    "cls_initializers": "",

    "cls_criterion": "Huber1Criterion",

    "cls_optimizers": "SGDOptimizer",
    "{cls_optimizers#0}.lr": 0.01,
    "{cls_optimizers#0}.momentum": 0.9,

    # "cls_schedulers": "ConstantScheduler",
    "cls_schedulers": "CosineScheduler",
    "{cls_schedulers#0}.warmup_epochs": 0,

    "cls_regularizers": "",
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
