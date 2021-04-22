from uninas.main import Main

# default configurations, for the search process and the network design
config_files = "{path_conf_bench_tasks}/s1_cotc_cifar.run_config, {path_conf_net_search}/bench201.run_config"


# these changes are applied to the default configuration in the config files
changes = {
    "{cls_task}.is_test_run": True,
    "{cls_task}.save_dir": "{path_tmp}/run_bench_cotc/",
    "{cls_task}.save_del_old": True,

    "{cls_trainer}.max_epochs": 4,
    "{cls_trainer}.eval_last": 0,
    "{cls_trainer}.test_last": 0,

    "{cls_method}.board_size": 3,
    "{cls_method}.grace_epochs": 0,
    "{cls_method}.match_batch_size": 8,
    "{cls_method}.mmn_batch_size": 16,

    "{cls_data}.dir": "{path_data}/cifar_data/",
    "{cls_data}.fake": False,
    "{cls_data}.download": False,
    "{cls_data}.batch_size_train": 96,

    # example how to mask options
    "{cls_method}.mask_indices": "0, 1, 4",                     # mask Zero, Skip, Pool
    "{cls_network_body}.cell_order": "n, n, r, n, n, r, n, n",  # 2 normal cells, one reduction cell, ...
    "{cls_network_stem}.features": 16,                          # start with 16 channels

    "{cls_schedulers#0}.warmup_epochs": 0,

    # specifying how to add weights, note that SplitWeightsMixedOp requires a SplitWeightsMixedOpCallback
    "{cls_network_cells_primitives#0}.mixed_cls": "MixedOp",  # MixedOp, BiasSumD1MixedOp, ...
    "{cls_network_cells_primitives#1}.mixed_cls": "MixedOp",  # MixedOp, BiasSumD1MixedOp, ...

    "cls_callbacks": "CheckpointCallback",
}


if __name__ == "__main__":
    task = Main.new_task(config_files, args_changes=changes)
    task.run()

