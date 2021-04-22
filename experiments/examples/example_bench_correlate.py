from uninas.main import Main


changes = {
    "cls_task": "EvalBenchTask",
    "{cls_task}.save_dir": "{path_tmp}/bench_correlate/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,
    "{cls_task}.measure_correlations": "'KendallTauNasMetric, PearsonNasMetric, SpearmanNasMetric'",

    # SIN s3
    "cls_benchmarks": "MiniNASTabularBenchmark, MiniNASTabularBenchmark, MiniNASTabularBenchmark, MiniNASTabularBenchmark, MiniNASTabularBenchmark, MiniNASTabularBenchmark",
    "{cls_benchmarks#0}.path": "{path_data}/bench/sin/SIN_fairnas_v0.1_only1.pt",
    "{cls_benchmarks#0}.default_result_type": "test",
    "{cls_benchmarks#0}.tmp_name": "(1)",
    "{cls_benchmarks#1}.path": "{path_data}/bench/sin/SIN_fairnas_v0.1_only1.pt",
    "{cls_benchmarks#1}.default_result_type": "train",
    "{cls_benchmarks#1}.tmp_name": "(1)",
    "{cls_benchmarks#2}.path": "{path_data}/bench/sin/SIN_fairnas_v0.1_only2.pt",
    "{cls_benchmarks#2}.default_result_type": "test",
    "{cls_benchmarks#2}.tmp_name": "(2)",
    "{cls_benchmarks#3}.path": "{path_data}/bench/sin/SIN_fairnas_v0.1_only2.pt",
    "{cls_benchmarks#3}.default_result_type": "train",
    "{cls_benchmarks#3}.tmp_name": "(2)",
    "{cls_benchmarks#4}.path": "{path_data}/bench/sin/SIN_fairnas_v0.1.pt",
    "{cls_benchmarks#4}.default_result_type": "test",
    "{cls_benchmarks#4}.tmp_name": "(1+2)",
    "{cls_benchmarks#5}.path": "{path_data}/bench/sin/SIN_fairnas_v0.1.pt",
    "{cls_benchmarks#5}.default_result_type": "train",
    "{cls_benchmarks#5}.tmp_name": "(1+2)",

    # SIN s3/s1, fully retrained 50 and one s1 network that evaluated 50+200
    # "cls_benchmarks": "MiniNASTabularBenchmark, MiniNASTabularBenchmark",
    # "{cls_benchmarks#0}.path": "{path_data}/bench/sin/SIN_fairnas_v0.1.pt",
    # "{cls_benchmarks#0}.default_result_type": "test",
    # "{cls_benchmarks#0}.tmp_name": "(1)",
    # "{cls_benchmarks#1}.path": "{path_data}/bench/sin/SIN_fairnas_eval250_382342.pt",
    # "{cls_benchmarks#1}.default_result_type": "test",
    # "{cls_benchmarks#1}.tmp_name": "(1)",

    # nats vs nats
    # "cls_benchmarks": "MiniNASTabularBenchmark, MiniNASTabularBenchmark, MiniNASTabularBenchmark",
    # "{cls_benchmarks#0}.path": "{path_data}/bench/nats/nats_bench_1.1_mini.pt",
    # "{cls_benchmarks#0}.default_result_type": "train",
    # "{cls_benchmarks#0}.tmp_name": "NATS",
    # "{cls_benchmarks#1}.path": "{path_data}/bench/nats/nats_bench_1.1_mini.pt",
    # "{cls_benchmarks#1}.default_result_type": "valid",
    # "{cls_benchmarks#1}.tmp_name": "NATS",
    # "{cls_benchmarks#2}.path": "{path_data}/bench/nats/nats_bench_1.1_mini.pt",
    # "{cls_benchmarks#2}.default_result_type": "test",
    # "{cls_benchmarks#2}.tmp_name": "NATS",

    # nats vs s3 runs
    # "cls_benchmarks": "MiniNASTabularBenchmark, MiniNASTabularBenchmark, MiniNASTabularBenchmark",
    # "{cls_benchmarks#0}.path": "{path_data}/bench/sin/bench201_n5c16_1.pt",
    # "{cls_benchmarks#0}.default_result_type": "train",
    # "{cls_benchmarks#0}.tmp_name": "(1)",
    # "{cls_benchmarks#1}.path": "{path_data}/bench/sin/bench201_n5c16_1.pt",
    # "{cls_benchmarks#1}.default_result_type": "test",
    # "{cls_benchmarks#1}.tmp_name": "(1)",
    # "{cls_benchmarks#2}.path": "{path_data}/bench/bench/sin.1_mini.pt",
    # "{cls_benchmarks#2}.default_result_type": "test",
    # "{cls_benchmarks#2}.tmp_name": "NATS",
}


if __name__ == "__main__":
    task = Main.new_task([], args_changes=changes)
    task.run()
