from uninas.main import Main

# pure HPO on the bench
# config_files = "{path_conf_bench_tasks}/s2_hpo_nsga2.run_config"
# config_files = "{path_conf_bench_tasks}/s2_hpo_random.run_config"
# config_files = "{path_conf_bench_tasks}/s2_pymoo_nsga2.run_config"

# HPO with an evaluation network
config_files = "{path_conf_bench_tasks}/s2_eval_net_bench.run_config"


changes = {
    "{cls_task}.is_test_run": True,
    "{cls_task}.save_dir": "{path_tmp}/run_bench_hpo/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.mini_bench_path": "{path_data}/nasbench201_1.1_mini.pt",

    # to be removed if you run a pure HPO on the bench
    "{cls_task}.s1_path": "{path_tmp}/run_bench_s1/",
    "{cls_task}.measure_top": "'10, 25, 50, 100, 150, 250, 500'",

    "{cls_task}.mini_bench_dataset": "cifar10",
    # "{cls_task}.mini_bench_dataset": "ImageNet16-120",
    # "{cls_task}.mini_bench_dataset": "cifar100",

    # "{cls_task}.mask_indices": "0, 1",
}


if __name__ == "__main__":
    task = Main.new_task(config_files, args_changes=changes)
    task.run()
