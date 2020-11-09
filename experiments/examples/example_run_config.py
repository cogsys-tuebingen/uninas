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
# config_files = "{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/test_small_net.run_config"
# config_files = "{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/DNU_scarletnas.run_config"
# config_files = "{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/efficientnet.run_config"
# config_files = "{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/fairnas.run_config"
# config_files = "{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/mobilenet_v2.run_config"
# config_files = "{path_conf_tasks}/s1_random.run_config, {path_conf_net_search}/proxylessnas.run_config"
# config_files = "{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/shufflenet_v2.run_config"
# config_files = "{path_conf_tasks}/s1_random.run_config, {path_conf_net_search}/uninas1.run_config"
# config_files = "{path_conf_tasks}/s2_self_nsga2.run_config"
# config_files = "{path_conf_tasks}/s2_pymoo_nsga2.run_config"
config_files = "{path_conf_tasks}/s3.run_config"
# config_files = "{path_conf_tasks}/s3rms.run_config"


changes = {
    "{cls_task}.is_test_run": True,
    "{cls_task}.save_dir": "{path_tmp}/run_config/",
    "{cls_task}.save_del_old": True,

    # "{cls_network}.config_path": "{path_conf_net_originals}/MobileNetV2.network_config",
    # "{cls_network}.config_path": "{path_conf_net_originals}/SPOSNet.network_config",

    # "cls_trainer": "LightningTrainer",  # SimpleTrainer, LightningTrainer
    "{cls_trainer}.max_epochs": 6,

    # "cls_data": "Cifar10Data",
    "{cls_data}.fake": True,
    "{cls_data}.dir": "{path_data}/ImageNet_ILSVRC2012/",
    "{cls_data}.batch_size_train": 2,
    "{cls_data}.batch_size_test": -1,
}


if __name__ == "__main__":
    # task = Main.new_task(config_files, args_changes=changes, raise_unparsed=False)
    task = Main.new_task(config_files, args_changes=changes)
    # task = Main.new_task(config_files)

    # print(task.methods[0].get_network())
    task.load()
    task.run()
