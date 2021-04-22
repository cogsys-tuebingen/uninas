from uninas.main import Main

"""
training a super net

beware that we are using fake data and a test run (only 10 steps/epoch)
and either run example_s3 with a fairnas model, or change the teacher
"""


# search network
config_files = "{path_conf_net_search}/fairnas.run_config"


args = {
    "cls_task": "SingleSearchTask",
    "{cls_task}.save_dir": "{path_tmp}/dna1/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    "cls_device": "CudaDevicesManager",
    "{cls_device}.num_devices": 1,

    "cls_trainer": "SimpleTrainer",
    "{cls_trainer}.max_epochs": 6,
    "{cls_trainer}.eval_last": 2,
    "{cls_trainer}.test_last": 2,

    "cls_exp_loggers": "TensorBoardExpLogger",
    "{cls_exp_loggers#0}.log_graph": False,

    "cls_callbacks": "CheckpointCallback",
    "{cls_callbacks#0}.top_n": 1,
    "{cls_callbacks#0}.key": "val/loss",
    "{cls_callbacks#0}.minimize_key": True,

    "cls_data": "Imagenet1000Data",
    "{cls_data}.fake": False,
    "{cls_data}.batch_size_train": 8,
    "{cls_data}.dir": "{path_data}/ImageNet_ILSVRC2012/",
    "{cls_data}.valid_split": 12800,
    "{cls_data}.valid_as_test": True,

    "cls_augmentations": "TimmImagenetAug",
    "{cls_augmentations#0}.crop_size": 224,

    "cls_method": "DnaMethod",
    "{cls_method}.mask_indices": "",

    "{cls_method}.teacher_cells_first": 1,
    "{cls_method}.teacher_cells_last": 6,
    "{cls_method}.split_by_features": True,
    "{cls_method}.loss_weights": "0.0684, 0.171, 0.3422, 0.2395, 0.5474, 0.3422",
    "{cls_method}.optimizer_lr_multipliers": "0.4, 1, 1, 1, 1, 0.4",
    "cls_teacher_network": "EfficientNetTimmNetwork",
    "{cls_network}.assert_output_match": True,
    "{cls_teacher_network}.model_name": "tf_efficientnet_b7",
    "{cls_teacher_network}.checkpoint_path": "{path_pretrained}",

    # "{cls_method}.teacher_cells_first": 0,
    # "{cls_method}.teacher_cells_last": 18,
    # "{cls_method}.split_by_features": True,
    # "cls_teacher_network": "RetrainUninasNetwork",
    # "{cls_network}.assert_output_match": True,
    # "{cls_teacher_network}.config_path": "FairNasC",
    # "{cls_method}.teacher_assert_trained": False,
    # "{cls_teacher_network}.checkpoint_path": "{path_tmp}/s3/",

    # "{cls_method}.teacher_cells_first": 1,
    # "{cls_method}.teacher_cells_last": 4,
    # "{cls_method}.split_by_features": False,
    # "cls_teacher_network": "BackboneMMDetNetwork",
    # "{cls_teacher_network}.assert_output_match": False,
    # "{cls_teacher_network}.config_path": "/home/laube/CodeGit/mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py",
    # "{cls_teacher_network}.checkpoint_path": "",
    # "{cls_teacher_network}.use_config_pretrained": True,

    "cls_strategy": "RandomChoiceStrategy",

    "cls_metrics": "DistillL2Metric",

    "cls_initializers": "SPOSInitializer",

    "cls_regularizers": "DropPathRegularizer",
    "{cls_regularizers#0}.min_prob": 0.2,
    "{cls_regularizers#0}.max_prob": 0.3,

    "cls_criterion": "DistillL2Criterion",

    "cls_optimizers": "SGDOptimizer",
    "{cls_optimizers#0}.lr": 0.05,
    "{cls_optimizers#0}.momentum": 0.9,
    "{cls_optimizers#0}.clip_norm_value": 5.0,
    "{cls_optimizers#0}.clip_norm_type": 2,

    "cls_schedulers": "ExponentialScheduler",
    "{cls_schedulers#0}.gamma": 0.9,
    "{cls_schedulers#0}.each_samples": 3000000,
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task(config_files, args_changes=args)
    # print(task.get_method().get_network())
    task.load()
    task.run()
