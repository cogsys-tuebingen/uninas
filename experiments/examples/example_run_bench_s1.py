from uninas.main import Main

config_files = "{path_conf_bench_tasks}/s1_fairnas_cifar.run_config, {path_conf_net_search}/bench201.run_config"
# config_files = "{path_conf_bench_tasks}/s1_random_cifar.run_config, {path_conf_net_search}/bench201.run_config"


# these changes are applied to the default configuration in the config files
changes = {
    "{cls_task}.is_test_run": True,
    "{cls_task}.save_dir": "{path_tmp}/run_bench_s1/",
    "{cls_task}.save_del_old": True,

    "{cls_trainer}.max_epochs": 4,

    "{cls_data}.dir": "{path_data}/cifar_data/",
    "{cls_data}.batch_size_train": 96,

    # example how to mask options
    "{cls_method}.mask_indices": "0, 1, 4",
    "{cls_network_body}.cell_order": "n, n, r, n, n, r, n, n",
    "{cls_network_stem}.features": 32,

    # some used augmentations
    "cls_augmentations": "DartsCifarAug",
    # "cls_augmentations": "AACifar10Aug",
    # "cls_augmentations": "DartsCifarAug, CutoutAug",

    "{cls_criterion}.smoothing_epsilon": 0.1,

    "{cls_schedulers#0}.warmup_epochs": 0,

    "{cls_network_cells_primitives#0}.mixed_cls": "VariableDepthMixedOp",
    "{cls_network_cells_primitives#1}.mixed_cls": "VariableDepthMixedOp",

    "cls_callbacks": "CheckpointCallback, VariableDepthMixedOpCallback",
    "{cls_callbacks#1}.milestones": "2",
    "{cls_callbacks#1}.pattern": "0, 1, 0",
}


if __name__ == "__main__":
    task = Main.new_task(config_files, args_changes=changes)
    task.run()
