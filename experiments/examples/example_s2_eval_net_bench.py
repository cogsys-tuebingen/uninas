from uninas.main import Main

"""
searching the best architecture in a super net
run example_run_bench_s1.py first
most arguments are taken from the s1's run_config, e.g. network design, metrics, trainer, ...

beware that s1 may be using fake data
"""


args = {
    "cls_task": "EvalNetBenchTask",
    # "{cls_task}.s1_path": "{path_tmp}/run_bench_s1/",
    "{cls_task}.s1_path": "{path_tmp}/run_config/",

    "{cls_task}.save_dir": "{path_tmp}/s2/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,
    "{cls_task}.measure_top": "",
    "{cls_task}.measure_correlations": "'KendallTauCorrelation, PearsonCorrelation, SpearmanCorrelation'",

    "cls_benchmark": "MiniNASTabularBenchmark",
    "{cls_benchmark}.path": "{path_data}/generated_bench/SIN_fairnas_mini_only1.pt",
    # "{cls_benchmark}.path": "{path_data}/nats_bench_1.1_mini.pt",
    # "{cls_benchmark}.default_data_set": "cifar100",

    "cls_hpo_self_algorithm": "RandomHPO",
    "{cls_hpo_self_algorithm}.num_eval": 10,

    "{cls_data}.batch_size_train": 16,

    "cls_hpo_estimators": "NetValueEstimator",
    "{cls_hpo_estimators#0}.key": "acc1",
    "{cls_hpo_estimators#0}.is_constraint": False,
    "{cls_hpo_estimators#0}.is_objective": True,
    "{cls_hpo_estimators#0}.maximize": True,
    "{cls_hpo_estimators#0}.load": False,
    "{cls_hpo_estimators#0}.batches_forward": 0,
    "{cls_hpo_estimators#0}.batches_train": 10,
    "{cls_hpo_estimators#0}.batches_eval": -1,
    "{cls_hpo_estimators#0}.value": "val/accuracy/1",
}


if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
