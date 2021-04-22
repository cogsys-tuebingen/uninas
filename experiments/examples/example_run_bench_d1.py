from uninas.main import Main

config_files = "{path_conf_tasks}/d1_dartsv1.run_config, {path_conf_net_search}/bench201.run_config"
# config_files = "{path_conf_tasks}/d1_asap.run_config, {path_conf_net_search}/bench201.run_config"
# config_files = "{path_conf_tasks}/d1_gdas.run_config, {path_conf_net_search}/bench201.run_config"
# config_files = "{path_conf_tasks}/d1_mdenas.run_config, {path_conf_net_search}/bench201.run_config"

changes = {
    "{cls_task}.is_test_run": True,
    "{cls_task}.save_dir": "{path_tmp}/run_bench_d1/",
    "{cls_task}.save_del_old": True,

    "cls_benchmark": "MiniNASTabularBenchmark",
    "{cls_benchmark}.path": "{path_data}/bench/nats/nats_bench_1.1_mini.pt",
    "{cls_benchmark}.default_data_set": "cifar100",
    "{cls_benchmark}.default_result_type": "test",

    "{cls_trainer}.max_epochs": 4,

    # "cls_data": "Cifar10Data",
    "{cls_data}.fake": True,
    "{cls_data}.dir": "{path_data}/ImageNet_ILSVRC2012/",
    "{cls_data}.batch_size_train": 2,
    "{cls_data}.batch_size_test": -1,
}


if __name__ == "__main__":
    task = Main.new_task(config_files, args_changes=changes)
    task.load()
    task.run()
