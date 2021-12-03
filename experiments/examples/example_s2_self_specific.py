from uninas.main import Main

"""
searching the best architecture in a super net
run example_s1.py first
most arguments are taken from the supernet1's run_config, e.g. network design, metrics, trainer, ...

beware that s1 is using fake data
"""


args = {
    "cls_task": "NetHPOTask",
    "{cls_task}.s1_path": "{path_tmp}/s1/",  # run_config, s1

    "{cls_task}.save_dir": "{path_tmp}/s2/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    "cls_device": "CudaDevicesManager",
    "{cls_device}.num_devices": 1,

    "cls_hpo_self_algorithm": "SpecificHPO",
    "{cls_hpo_self_algorithm}.values": "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;"
                                       "0, 0, 3, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;",

    "{cls_data}.dir": "{path_data}/ImageNet_ILSVRC2012/",
    "{cls_data}.fake": True,
    "{cls_data}.batch_size_train": 2,

    "cls_hpo_estimators": "NetMacsEstimator, NetValueEstimator, NetParamsEstimator",
    # macs
    "{cls_hpo_estimators#0}.key": "macs",
    "{cls_hpo_estimators#0}.is_constraint": True,
    "{cls_hpo_estimators#0}.min_value": 0.0,
    "{cls_hpo_estimators#0}.max_value": 330*10e9,
    "{cls_hpo_estimators#0}.is_objective": True,
    "{cls_hpo_estimators#0}.maximize": False,
    # accuracy
    "{cls_hpo_estimators#1}.key": "accuracy",
    "{cls_hpo_estimators#1}.is_constraint": False,
    "{cls_hpo_estimators#1}.is_objective": True,
    "{cls_hpo_estimators#1}.maximize": True,
    "{cls_hpo_estimators#1}.load": False,
    "{cls_hpo_estimators#1}.batches_forward": 0,
    "{cls_hpo_estimators#1}.batches_train": 0,
    "{cls_hpo_estimators#1}.batches_eval": -1,
    "{cls_hpo_estimators#1}.value": "val/accuracy/1",
    # params
    "{cls_hpo_estimators#2}.key": "params",
    "{cls_hpo_estimators#2}.is_constraint": False,
    "{cls_hpo_estimators#2}.is_objective": True,
    "{cls_hpo_estimators#2}.maximize": False,
    "{cls_hpo_estimators#2}.count_only_trainable": True,
}


if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
