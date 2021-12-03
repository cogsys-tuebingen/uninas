from uninas.main import Main

# default configurations, for the search process and the network design
config_files = "{path_conf_tasks}/s1_random.run_config, {path_conf_net_search}/efficientnet.run_config"


# these changes are applied to the default configuration in the config files
changes = {
    # this is a test run! every epoch ends after 10 training steps
    "{cls_task}.is_test_run": True,
    "{cls_task}.save_dir": "{path_tmp}/demo/icw/train_supernet/",
    "{cls_task}.save_del_old": True,

    # only 4 epochs to test
    "{cls_trainer}.max_epochs": 4,

    "{cls_schedulers#0}.warmup_epochs": 0,

    "cls_initializers": "",

    "cls_network_cells_primitives": "EfficientNetPrimitivesMini, EfficientNetPrimitivesMini, EfficientNetPrimitivesMini, EfficientNetPrimitivesMini, EfficientNetPrimitivesMini, EfficientNetPrimitivesMini, EfficientNetPrimitivesMini",

    # data, small batch size to test
    "cls_data": "SubImagenetc100t1000v500Data",
    "{cls_data}.dir": "{path_data}/SubImageNet_c100_t1000_v500/",
    "{cls_data}.fake": False,
    "{cls_data}.download": False,
    "{cls_data}.batch_size_train": 32,
    "{cls_data}.valid_split": 0.1,

    # weaker augmentations worked better
    "cls_augmentations": "DartsImagenetAug",
    # "{cls_augmentations#0}.color_jitter": 0.0,
    # "{cls_augmentations#0}.scale_min": 0.4,
    # "{cls_augmentations#0}.interpolation": "bilinear",

    # specifying how to add weights, note that SplitWeightsMixedOp requires a SplitWeightsMixedOpCallback
    "{cls_network_cells_primitives#0}.mixed_cls": "SplitWeightsMixedOp",  # MixedOp, SplitWeightsMixedOp, ...
    "{cls_network_cells_primitives#1}.mixed_cls": "SplitWeightsMixedOp",  # MixedOp, SplitWeightsMixedOp, ...
    "{cls_network_cells_primitives#2}.mixed_cls": "SplitWeightsMixedOp",  # MixedOp, SplitWeightsMixedOp, ...
    "{cls_network_cells_primitives#3}.mixed_cls": "SplitWeightsMixedOp",  # MixedOp, SplitWeightsMixedOp, ...
    "{cls_network_cells_primitives#4}.mixed_cls": "SplitWeightsMixedOp",  # MixedOp, SplitWeightsMixedOp, ...
    "{cls_network_cells_primitives#5}.mixed_cls": "SplitWeightsMixedOp",  # MixedOp, SplitWeightsMixedOp, ...
    "{cls_network_cells_primitives#6}.mixed_cls": "SplitWeightsMixedOp",  # MixedOp, SplitWeightsMixedOp, ...

    # add a SplitWeightsMixedOpCallback and corresponding arguments if you use SplitWeightsMixedOp
    # "cls_callbacks": "CheckpointCallback",
    "cls_callbacks": "CheckpointCallback, SplitWeightsMixedOpCallback",
    "{cls_callbacks#1}.milestones": "3",    # split after 2 epochs
    "{cls_callbacks#1}.pattern": "1",       # split every SplitWeightsMixedOp
}


if __name__ == "__main__":
    task = Main.new_task(config_files, args_changes=changes)
    task.run()
