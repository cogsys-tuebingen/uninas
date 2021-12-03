from uninas.main import Main

"""
create a bench from a s1 net
run example_run_bench_s1.py first
most arguments are taken from the s1's run_config, e.g. network design, metrics, trainer, ...

beware that s1 may be using fake data
"""


args = {
    "cls_task": "CreateSearchNetBenchTask",
    "{cls_task}.s1_path": "{path_tmp}/run_bench_s1/",

    "{cls_task}.save_dir": "{path_tmp}/s2_bench/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,
    "{cls_task}.measure_min": 20,

    "cls_benchmarks": "MiniNASTabularBenchmark",
    # "{cls_benchmarks#0}.path": "{path_data}/bench/sin/SIN_fairnas_v0.1.pt",
    "{cls_benchmarks#0}.path": "{path_data}/bench/nats/nats_bench_1.1_mini.pt",

    "cls_hpo_self_algorithm": "RandomHPO",
    "{cls_hpo_self_algorithm}.num_eval": 20,

    "{cls_data}.batch_size_train": 16,

    "cls_hpo_estimators": "NetValueEstimator, NetMacsEstimator",
    "{cls_hpo_estimators#0}.key": "acc1/valid",
    "{cls_hpo_estimators#0}.is_objective": True,
    "{cls_hpo_estimators#0}.load": False,
    "{cls_hpo_estimators#0}.batches_forward": 0,
    "{cls_hpo_estimators#0}.batches_train": 5,
    "{cls_hpo_estimators#0}.batches_eval": -1,
    "{cls_hpo_estimators#0}.value": "val/accuracy/1",
    "{cls_hpo_estimators#1}.key": "flops",
    "{cls_hpo_estimators#1}.is_objective": True,
}


if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
