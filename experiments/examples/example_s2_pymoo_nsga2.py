from uninas.main import Main

"""
searching the best architecture in a super net
run example_s1.py first
most arguments are taken from the supernet1's run_config, e.g. network design, metrics, trainer, ...

beware that s1 is using fake data
"""


args = {
    "cls_task": "NetPymooHPOTask",
    "{cls_task}.s1_path": "{path_tmp}/s1/",
    # "{cls_task}.s1_path": "{path_tmp}/from_config/",

    "{cls_task}.save_dir": "{path_tmp}/s2/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    "cls_device": "CudaDevicesManager",
    "{cls_device}.num_devices": 1,

    "cls_hpo_pymoo_algorithm": "NSGA2PymooAlgorithm",
    "{cls_hpo_pymoo_algorithm}.pop_size": 10,
    "{cls_hpo_pymoo_algorithm}.n_offsprings": 5,

    "cls_hpo_estimators": "NetMacsEstimator, NetValueEstimator",
    # macs
    "{cls_hpo_estimators#0}.key": 'macs',
    "{cls_hpo_estimators#0}.is_constraint": True,
    "{cls_hpo_estimators#0}.min_value": 0.0,
    "{cls_hpo_estimators#0}.max_value": 330000000,
    "{cls_hpo_estimators#0}.is_objective": True,
    "{cls_hpo_estimators#0}.maximize": False,
    # accuracy
    "{cls_hpo_estimators#1}.key": 'acc1',
    "{cls_hpo_estimators#1}.is_constraint": False,
    "{cls_hpo_estimators#1}.is_objective": True,
    "{cls_hpo_estimators#1}.maximize": True,
    "{cls_hpo_estimators#1}.load": False,
    "{cls_hpo_estimators#1}.batches_forward": 0,
    "{cls_hpo_estimators#1}.batches_train": 0,
    "{cls_hpo_estimators#1}.batches_eval": 5,
    "{cls_hpo_estimators#1}.value": "val/accuracy/1",

    "cls_hpo_pymoo_termination": "NIterPymooTermination",
    "{cls_hpo_pymoo_termination}.n": 4,

    "cls_hpo_pymoo_sampler": "IntRandomPymooSampler",

    "cls_hpo_pymoo_crossover": "SbxPymooCrossover",
    "{cls_hpo_pymoo_crossover}.type": "int",
    "{cls_hpo_pymoo_crossover}.prob": 0.9,
    "{cls_hpo_pymoo_crossover}.eta": 15,

    "cls_hpo_pymoo_mutation": "PolynomialPymooMutation",
    "{cls_hpo_pymoo_mutation}.type": "int",
    "{cls_hpo_pymoo_mutation}.eta": 20
}


if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
