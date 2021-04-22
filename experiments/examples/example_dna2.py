from uninas.main import Main

"""
searching the best architecture in a super net
run example_s1.py first
most arguments are taken from the supernet1's run_config, e.g. network design, metrics, trainer, ...

beware that s1 is using fake data
"""


args = {
    "cls_task": "DnaHPOTask",
    "{cls_task}.s1_path": "{path_tmp}/dna1/",
    # "{cls_task}.s1_path": "{path_tmp}/from_config/",

    "{cls_task}.save_dir": "{path_tmp}/dna2/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    "{cls_task}.dna2_max_eval_per_stage": 5,
    "{cls_task}.dna2_max_eval_final": 20,

    "cls_device": "CudaDevicesManager",
    "{cls_device}.num_devices": 1,

    "{cls_data}.dir": "{path_data}/ImageNet_ILSVRC2012/",
    "{cls_data}.fake": False,
    "{cls_data}.batch_size_train": 32,

    "cls_hpo_estimators": "NetMacsEstimator, NetValueEstimator",
    # macs
    "{cls_hpo_estimators#0}.key": "macs",
    "{cls_hpo_estimators#0}.is_constraint": True,
    "{cls_hpo_estimators#0}.min_value": 0.0,
    "{cls_hpo_estimators#0}.max_value": 330*10e9,
    "{cls_hpo_estimators#0}.is_objective": True,
    "{cls_hpo_estimators#0}.maximize": False,
    # accuracy
    "{cls_hpo_estimators#1}.key": "loss",
    "{cls_hpo_estimators#1}.is_constraint": False,
    "{cls_hpo_estimators#1}.is_objective": True,
    "{cls_hpo_estimators#1}.maximize": False,
    "{cls_hpo_estimators#1}.load": False,
    "{cls_hpo_estimators#1}.batches_forward": 0,
    "{cls_hpo_estimators#1}.batches_train": 0,
    "{cls_hpo_estimators#1}.batches_eval": -1,
    "{cls_hpo_estimators#1}.value": "val/loss",
}


if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
