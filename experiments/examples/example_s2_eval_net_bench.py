from uninas.main import Main

"""
searching the best architecture in a super net
run example_run_bench_s1.py first
most arguments are taken from the s1's run_config, e.g. network design, metrics, masked operations, ...

beware that s1 may be using fake data
"""


args = {
    # evaluate a super-network on a bench
    "cls_task": "EvalNetBenchTask",
    "{cls_task}.save_dir": "{path_tmp}/s2/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    # where to find the trained super network
    "{cls_task}.s1_path": "{path_tmp}/run_bench_s1/",
    # "{cls_task}.s1_path": "{path_tmp}/s1/",
    # "{cls_task}.s1_path": "{path_tmp}/run_config/",

    # in addition to random networks, also pick the n best networks in the search space and evaluate them separately
    "{cls_task}.measure_top": 10,

    # which metrics to evaluate on all evaluation result
    "{cls_task}.nas_metrics": "'ImprovementNasMetric, KendallTauNasMetric, ByTargetsNasMetric'",

    # run against bench results
    "cls_benchmark": "MiniNASTabularBenchmark",
    "{cls_benchmark}.path": "{path_data}/bench/nats/nats_bench_1.1_mini.pt",
    # "{cls_benchmark}.path": "{path_data}/bench/sin/SIN_fairnas_v0.1.pt",
    # "{cls_benchmark}.default_data_set": "cifar100",

    # but evaluate at most this many networks
    "cls_hpo_self_algorithm": "RandomHPO",
    "{cls_hpo_self_algorithm}.num_eval": 10,

    # what and how to evaluate each specific network
    "cls_hpo_estimators": "NetValueEstimator",
    "{cls_hpo_estimators#0}.key": "acc1",
    "{cls_hpo_estimators#0}.is_constraint": False,
    "{cls_hpo_estimators#0}.is_objective": True,
    "{cls_hpo_estimators#0}.maximize": True,
    "{cls_hpo_estimators#0}.load": True,
    "{cls_hpo_estimators#0}.batches_forward": 20,
    "{cls_hpo_estimators#0}.batches_train": 0,
    "{cls_hpo_estimators#0}.batches_eval": -1,
    "{cls_hpo_estimators#0}.value": "val/accuracy/1",
}


if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.run()
