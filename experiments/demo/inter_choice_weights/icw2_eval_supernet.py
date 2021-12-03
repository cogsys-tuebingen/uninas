from uninas.main import Main

"""
searching the best architecture in a super net
run demo_train_supernet_nats.py first
most arguments are taken from the super-network's run_config, e.g. network design, metrics, masked operations, ...
"""


config_files = "{path_conf_bench_tasks}/s2_eval_net_bench.run_config"


changes = {
    "{cls_task}.save_dir": "{path_tmp}/demo/icw/eval_supernet/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    # where to find the trained super network, used to figure out the task settings
    "{cls_task}.s1_path": "{path_tmp}/demo/icw/train_supernet/",

    # in addition to random networks, also pick the n best networks in the search space and evaluate them separately
    "{cls_task}.measure_top": 10,

    # run against bench results
    "{cls_benchmark}.path": "{path_data}/bench/nats/nats_bench_1.1_mini.pt",
    # "{cls_benchmark}.path": "{path_data}/bench/sin/SIN_fairnas_v0.1.pt",
    # "{cls_benchmark}.default_data_set": "cifar100",

    # but evaluate at most this many networks
    "{cls_hpo_self_algorithm}.num_eval": 10,
}


if __name__ == "__main__":
    task = Main.new_task(config_files, args_changes=changes)
    task.run()
