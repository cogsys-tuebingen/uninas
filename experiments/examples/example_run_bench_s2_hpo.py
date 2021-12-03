from uninas.main import Main

# pure HPO on the bench
# config_files = "{path_conf_bench_tasks}/s2_hpo_nsga2.run_config"
# config_files = "{path_conf_bench_tasks}/s2_hpo_random.run_config"
config_files = "{path_conf_bench_tasks}/s2_pymoo_nsga2.run_config"

# HPO with an evaluation network
# config_files = "{path_conf_bench_tasks}/s2_eval_net_bench.run_config"


changes = {
    "{cls_task}.is_test_run": True,
    "{cls_task}.save_dir": "{path_tmp}/run_bench_hpo/",
    "{cls_task}.save_del_old": True,

    "cls_benchmark": "MiniNASTabularBenchmark",
    "{cls_benchmark}.default_data_set": "ImageNet16-120",  # cifar10, cifar100, ImageNet16-120
    "{cls_benchmark}.path": "{path_data}/bench/nats/nats_bench_1.1_mini.pt",
    "{cls_benchmark}.default_result_type": "test",

    # to be removed if you run a pure HPO on the bench
    # "{cls_task}.s1_path": "{path_tmp}/run_bench_s1/",
    # "{cls_task}.measure_top": 20,

    # "{cls_hpo_self_algorithm}.num_eval": 20,

    # "{cls_task}.mask_indices": "0, 1",

    # "{cls_data}.dir": "{path_data}/cifar_data/",

    # "{cls_hpo_estimators#0}.batches_forward": 5,

    "NIterPymooTermination.n": 10,
}


if __name__ == "__main__":
    task = Main.new_task(config_files, args_changes=changes)
    task.run()
