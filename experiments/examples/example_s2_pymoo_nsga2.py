from uninas.main import Main

"""
searching the best architecture in a super net
run example_s1.py or run_config.py first, whichever s1_path is set
most arguments are taken from the supernet1's run_config, e.g. network design, metrics, trainer, ...

beware that s1 is using fake data
"""


args = {
    "cls_task": "NetPymooHPOTask",
    "{cls_task}.s1_path": "{path_tmp}/s1/",  # run_config, s1

    "{cls_task}.save_dir": "{path_tmp}/s2/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    "cls_device": "CudaDevicesManager",
    "{cls_device}.num_devices": 1,

    "cls_hpo_pymoo_algorithm": "NSGA2PymooAlgorithm",
    "{cls_hpo_pymoo_algorithm}.pop_size": 10,
    "{cls_hpo_pymoo_algorithm}.n_offsprings": 5,

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

est = 1
# measure net directly
if est == 0:
    args.update({
        "cls_hpo_estimators": "NetValueEstimator, NetMacsEstimator",
        # accuracy, measured by forward passes
        "{cls_hpo_estimators#0}.key": "accuracy",
        "{cls_hpo_estimators#0}.is_constraint": False,
        "{cls_hpo_estimators#0}.is_objective": True,
        "{cls_hpo_estimators#0}.maximize": True,
        "{cls_hpo_estimators#0}.load": False,
        "{cls_hpo_estimators#0}.batches_forward": 0,
        "{cls_hpo_estimators#0}.batches_train": 0,
        "{cls_hpo_estimators#0}.batches_eval": -1,
        "{cls_hpo_estimators#0}.value": "val/accuracy/1",
        # macs
        "{cls_hpo_estimators#1}.key": "macs",
        "{cls_hpo_estimators#1}.is_constraint": True,
        "{cls_hpo_estimators#1}.min_value": 0.0,
        "{cls_hpo_estimators#1}.max_value": 330*10e9,
        "{cls_hpo_estimators#1}.is_objective": True,
        "{cls_hpo_estimators#1}.maximize": False,
    })
# use trained profiler for latency/macs
if est == 1:
    args.update({
        # "cls_hpo_estimators": "NetValueEstimator, NetMacsEstimator",
        "cls_hpo_estimators": "NetValueEstimator, ModelEstimator, ModelEstimator",
        # accuracy, measured by forward passes
        "{cls_hpo_estimators#0}.key": "loss",
        "{cls_hpo_estimators#0}.is_constraint": False,
        "{cls_hpo_estimators#0}.is_objective": True,
        "{cls_hpo_estimators#0}.maximize": False,
        "{cls_hpo_estimators#0}.load": False,
        "{cls_hpo_estimators#0}.batches_forward": 0,
        "{cls_hpo_estimators#0}.batches_train": 0,
        "{cls_hpo_estimators#0}.batches_eval": -1,
        "{cls_hpo_estimators#0}.value": "val/loss",
        # # profiled macs
        "{cls_hpo_estimators#1}.key": "macs_model",
        "{cls_hpo_estimators#1}.is_objective": True,
        "{cls_hpo_estimators#1}.maximize": False,
        "{cls_hpo_estimators#1}.model_file_path": '{path_profiled}/tab_fairnas_macs.pt',
        "{cls_hpo_estimators#1}.cast_one_hot": False,
        # # profiled latency
        "{cls_hpo_estimators#2}.key": "latency_model",
        "{cls_hpo_estimators#2}.is_objective": True,
        "{cls_hpo_estimators#2}.maximize": False,
        # "{cls_hpo_estimators#2}.model_file_path": '{path_profiled}/tab_fairnas_latency.pt',
        # "{cls_hpo_estimators#2}.model_file_path": '{path_profiled}/rf_fairnas_latency_cpu.pt',
        # "{cls_hpo_estimators#2}.model_file_path": '{path_profiled}/xgb_fairnas_latency_cpu_oh.pt',
        "{cls_hpo_estimators#2}.model_file_path": '{path_profiled}/nn_fairnas_latency_cpu_oh_test.pt',
        "{cls_hpo_estimators#2}.model_device": 'cuda:0',
        "{cls_hpo_estimators#2}.cast_one_hot": True,
    })


if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
