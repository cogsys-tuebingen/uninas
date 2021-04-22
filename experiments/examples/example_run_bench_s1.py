from uninas.main import Main

# default configurations, for the search process and the network design
# config_files = "{path_conf_bench_tasks}/s1_fairnas_cifar.run_config, {path_conf_net_search}/bench201.run_config"
config_files = "{path_conf_bench_tasks}/s1_random_cifar.run_config, {path_conf_net_search}/bench201.run_config"


# these changes are applied to the default configuration in the config files
changes = {
    "{cls_task}.is_test_run": True,
    "{cls_task}.save_dir": "{path_tmp}/run_bench_s1/",
    "{cls_task}.save_del_old": True,

    "{cls_trainer}.max_epochs": 4,

    "{cls_data}.dir": "{path_data}/cifar_data/",
    "{cls_data}.fake": False,
    "{cls_data}.download": False,
    "{cls_data}.batch_size_train": 96,

    # example how to mask options
    "{cls_method}.mask_indices": "0, 1, 4",                     # ops={Zero, Skip, 1x1, 3x3, Pool}
    "{cls_network_body}.cell_order": "n, n, r, n, n, r, n, n",  # 2 normal cells, one reduction cell, ...
    "{cls_network_stem}.features": 16,                          # start with 16 channels

    # some augmentations
    "cls_augmentations": "DartsCifarAug",                       # default augmentations for cifar
    # "cls_augmentations": "AACifar10Aug",
    # "cls_augmentations": "DartsCifarAug, CutoutAug",

    "{cls_schedulers#0}.warmup_epochs": 0,

    # specifying how to add weights, note that SplitWeightsMixedOp requires a SplitWeightsMixedOpCallback
    "{cls_network_cells_primitives#0}.mixed_cls": "SplitWeightsMixedOp",  # MixedOp, SplitWeightsMixedOp, BiasSumD1MixedOp, ...
    "{cls_network_cells_primitives#1}.mixed_cls": "SplitWeightsMixedOp",  # MixedOp, SplitWeightsMixedOp, BiasSumD1MixedOp, ...

    "cls_callbacks": "CheckpointCallback, SplitWeightsMixedOpCallback",
    "{cls_callbacks#1}.milestones": "2",    # split after 2 epochs
    "{cls_callbacks#1}.pattern": "1",       # split every SplitWeightsMixedOp
}


if __name__ == "__main__":
    task = Main.new_task(config_files, args_changes=changes)
    task.run()
