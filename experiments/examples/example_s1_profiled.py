from uninas.main import Main

"""
training a super net
"""

config_files = "{path_conf_tasks}/dsnas_hwnas.run_config, {path_conf_net_search}/uninas_learn_predictors.run_config"


args = {
    "{cls_task}.is_test_run": True,
    "{cls_task}.save_del_old": True,
    "{cls_task}.save_dir": "{path_tmp}/fit/s1/",
    "{cls_trainer}.max_epochs": 20,

    "cls_data": "ProfiledData",
    "{cls_data}.valid_split": 0.0,
    "{cls_data}.batch_size_train": 200,
    # "{cls_data}.dir": "{path_data}/profiling/fairnas/cpu/",
    # "{cls_data}.file_name": "data_standalone.pt",
    # "{cls_data}.file_name": "data_overcomplete.pt",
    "{cls_data}.dir": "{path_data}/profiling/HW-NAS/",
    "{cls_data}.file_name": "cifar10-edgegpu_energy.pt",
    # "{cls_data}.file_name": "ImageNet16-120-raspi4_latency.pt",
    "{cls_data}.cast_one_hot": True,
    "{cls_data}.normalize_labels": True,
    "{cls_data}.train_num": -1,

    "cls_method": "DSNASMethod",  # StrictlyFairRandomMethod, UniformRandomMethod, DSNASMethod
    "{cls_method}.mask_indices": "",

    "cls_criterion": "HuberCriterion",

    "{cls_optimizers#0}.lr": 0.01,
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task(config_files, args_changes=args)
    # print(task.get_method().get_network())
    task.load()
    task.run()
