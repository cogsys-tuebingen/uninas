from uninas.main import Main


changes = {
    "cls_task": "EvalBenchTask",
    "{cls_task}.save_dir": "{path_tmp}/bench_correlate/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,
    "{cls_task}.measure_correlations": "'KendallTauCorrelation, PearsonCorrelation, SpearmanCorrelation'",

    # "cls_benchmarks": "MiniNASTabularBenchmark, MiniNASTabularBenchmark, MiniNASTabularBenchmark, MiniNASTabularBenchmark",
    # "{cls_benchmarks#0}.path": "{path_data}/generated_bench/SIN_fairnas_mini_only1.pt",
    # "{cls_benchmarks#0}.default_result_type": "test",
    # "{cls_benchmarks#1}.path": "{path_data}/generated_bench/SIN_fairnas_mini_only1.pt",
    # "{cls_benchmarks#1}.default_result_type": "train",
    # "{cls_benchmarks#2}.path": "{path_data}/generated_bench/SIN_fairnas_mini_only2.pt",
    # "{cls_benchmarks#2}.default_result_type": "test",
    # "{cls_benchmarks#3}.path": "{path_data}/generated_bench/SIN_fairnas_mini_only2.pt",
    # "{cls_benchmarks#3}.default_result_type": "train",

    "cls_benchmarks": "MiniNASTabularBenchmark, MiniNASTabularBenchmark, MiniNASTabularBenchmark",
    "{cls_benchmarks#0}.path": "{path_data}/nats_bench_1.1_mini.pt",
    "{cls_benchmarks#0}.default_result_type": "train",
    "{cls_benchmarks#1}.path": "{path_data}/nats_bench_1.1_mini.pt",
    "{cls_benchmarks#1}.default_result_type": "valid",
    "{cls_benchmarks#2}.path": "{path_data}/nats_bench_1.1_mini.pt",
    "{cls_benchmarks#2}.default_result_type": "test",

    # "cls_benchmarks": "MiniNASTabularBenchmark, MiniNASTabularBenchmark, MiniNASTabularBenchmark",
    # "{cls_benchmarks#0}.path": "{path_data}/generated_bench/bench201_n5c16_1.pt",
    # "{cls_benchmarks#0}.default_result_type": "train",
    # "{cls_benchmarks#1}.path": "{path_data}/generated_bench/bench201_n5c16_1.pt",
    # "{cls_benchmarks#1}.default_result_type": "test",
    # "{cls_benchmarks#2}.path": "{path_data}/nats_bench_1.1_mini.pt",
    # "{cls_benchmarks#2}.default_result_type": "test",
}


if __name__ == "__main__":
    task = Main.new_task([], args_changes=changes)
    task.run()
