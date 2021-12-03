from uninas.main import Main

"""
use an existing run_config file,
make minor changes (e.g. dataset with same size, fewer cells and channels)

beware that we are using fake data
"""

# darts-like
# config_files = "{path_conf_tasks}/d1_dartsv1.run_config, {path_conf_net_search}/darts.run_config"
# config_files = "{path_conf_tasks}/d1_asap.run_config, {path_conf_net_search}/darts.run_config"
# config_files = "{path_conf_tasks}/d1_gdas.run_config, {path_conf_net_search}/darts.run_config"
# config_files = "{path_conf_tasks}/d1_mdenas.run_config, {path_conf_net_search}/darts.run_config"
# config_files = "{path_conf_tasks}/d2_cifar.run_config"
# config_files = "{path_conf_tasks}/d2_imagenet.run_config"

# single/multi path nets
# config_files = "{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/fairnas.run_config"
# config_files = "{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/fairnas_shared.run_config"
# config_files = "{path_conf_tasks}/s1_random.run_config, {path_conf_net_search}/fairnas_small.run_config"
# config_files = "{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/test_small_net.run_config"
# config_files = "{path_conf_tasks}/s1_random.run_config, {path_conf_net_search}/scarletnas.run_config"
# config_files = "{path_conf_tasks}/s1_random.run_config, {path_conf_net_search}/efficientnet.run_config"
# config_files = "{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/mobilenet_v2.run_config"
# config_files = "{path_conf_tasks}/s1_random.run_config, {path_conf_net_search}/proxylessnas.run_config"
# config_files = "{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/shufflenet_v2.run_config"
# config_files = "{path_conf_tasks}/s1_random.run_config, {path_conf_net_search}/uninas1.run_config"
# config_files = "{path_conf_tasks}/s2_self_nsga2.run_config"
# config_files = "{path_conf_tasks}/s2_pymoo_nsga2.run_config"
# config_files = "{path_conf_tasks}/s3.run_config"
# config_files = "{path_conf_tasks}/s3_extract_s1.run_config"
# config_files = "{path_conf_tasks}/s3rms.run_config"
# config_files = "{path_conf_tasks}/s3_labpal.run_config"

# distill
# config_files = "{path_conf_tasks}/dna1.run_config, {path_conf_net_search}/fairnas.run_config"
# config_files = "{path_conf_tasks}/dna1.run_config, {path_conf_net_search}/efficientnet.run_config"
# config_files = "{path_conf_tasks}/dna1.run_config, {path_conf_net_search}/uninas1.run_config"
# config_files = "{path_conf_tasks}/dna2.run_config"

# cream of the crop search
# config_files = "{path_conf_tasks}/cream1.run_config, {path_conf_net_search}/fairnas.run_config"

# search + retrain in one
config_files = "{path_conf_tasks}/dsnas_imagenet.run_config, {path_conf_net_search}/fairnas.run_config"

# profile
# config_files = "{path_conf_tasks}/profile_macs.run_config, {path_conf_net_search}/fairnas.run_config"

# further bench-related tasks
# config_files = "{path_conf_bench_tasks}/s2_create_net_bench.run_config"

# fit profiling data
# config_files = "{path_conf_tasks}/fit_classic.run_config"
# config_files = "{path_conf_tasks}/fit_net.run_config"


changes = {
    "{cls_task}.is_test_run": True,
    "{cls_task}.save_dir": "{path_tmp}/run_config/",
    "{cls_task}.save_del_old": True,

    # "{cls_network}.config_path": "MobileNetV2",
    # "{cls_network}.config_path": "SPOSNet",

    # "cls_trainer": "LightningTrainer",  # SimpleTrainer, LightningTrainer
    "{cls_trainer}.max_epochs": 6,

    # "cls_data": "Cifar10Data",
    "{cls_data}.fake": True,
    "{cls_data}.dir": "{path_data}/ImageNet_ILSVRC2012/",
    "{cls_data}.batch_size_train": 4,
    "{cls_data}.batch_size_test": -1,

    # "{cls_network_cells_primitives#0}.mixed_cls": "AttentionD3SigmoidMixedOp",  # MixedOp, AttentionD3SigmoidMixedOp
}


if __name__ == "__main__":
    # task = Main.new_task(config_files, args_changes=changes, raise_unparsed=False)
    task = Main.new_task(config_files, args_changes=changes)
    # task = Main.new_task(config_files)

    # print(task.get_method().get_network())
    task.load()
    task.run()
