"""
training a super-network and periodically evaluating its performance on bench architectures
a work in this direction exists: https://arxiv.org/abs/2001.01431
"""

from uninas.main import Main

# default configurations, for the search process and the network design
# config_files = "{path_conf_bench_tasks}/s1_fairnas_cifar.run_config, {path_conf_net_search}/bench201.run_config"
config_files = "{path_conf_bench_tasks}/s1_random_cifar.run_config, {path_conf_net_search}/bench201.run_config"


# these changes are applied to the default configuration in the config files
changes = {
    "{cls_task}.is_test_run": True,
    "{cls_task}.save_dir": "{path_tmp}/run_bench_s1_per/",
    "{cls_task}.save_del_old": True,

    "{cls_trainer}.max_epochs": 4,

    "{cls_data}.dir": "{path_data}/cifar_data/",
    "{cls_data}.fake": False,
    "{cls_data}.download": False,
    "{cls_data}.batch_size_train": 96,

    # example how to mask options
    "{cls_method}.mask_indices": "0, 1, 4",                     # mask Zero, Skip, Pool
    "{cls_network_body}.cell_order": "n, n, r, n, n, r, n, n",  # 2 normal cells, one reduction cell, ...
    "{cls_network_stem}.features": 16,                          # start with 16 channels

    # some augmentations
    "cls_augmentations": "DartsCifarAug",                       # default augmentations for cifar

    "{cls_schedulers#0}.warmup_epochs": 0,

    # specifying how to add weights, note that SplitWeightsMixedOp requires a SplitWeightsMixedOpCallback
    "{cls_network_cells_primitives#0}.mixed_cls": "MixedOp",  # MixedOp, BiasMixedOp, ...
    "{cls_network_cells_primitives#1}.mixed_cls": "MixedOp",  # MixedOp, BiasMixedOp, ...

    "cls_callbacks": "CheckpointCallback, CreateBenchCallback",
    "{cls_callbacks#1}.each_epochs": 1,
    "{cls_callbacks#1}.reset_bn": True,
    "{cls_callbacks#1}.benchmark_path": "{path_data}/bench/nats/nats_bench_1.1_subset_m_test.pt",

    # what and how to evaluate each specific network
    "cls_cb_objectives": "NetValueEstimator",
    "{cls_cb_objectives#0}.key": "acc1/valid",
    "{cls_cb_objectives#0}.is_constraint": False,
    "{cls_cb_objectives#0}.is_objective": True,
    "{cls_cb_objectives#0}.maximize": True,
    "{cls_cb_objectives#0}.load": True,
    "{cls_cb_objectives#0}.batches_forward": 20,
    "{cls_cb_objectives#0}.batches_train": 0,
    "{cls_cb_objectives#0}.batches_eval": -1,
    "{cls_cb_objectives#0}.value": "val/accuracy/1",
}


if __name__ == "__main__":
    task = Main.new_task(config_files, args_changes=changes)
    task.run()
