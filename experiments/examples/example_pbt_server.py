from uninas.main import Main

"""
hosting a Population-based-training (PBT) server
"""


args = {
    # the task is to profile
    "cls_task": "PbtServerTask",
    "{cls_task}.save_dir": "{path_tmp}/pbt_server/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": False,

    # this path must match the one in the clients' callbacks
    "{cls_task}.communication_file": "{path_tmp}/pbt_server/communication_uri",
    "{cls_task}.num_clients": 1,

    "cls_pbt_selector": "DefaultPbtSelector",
    "{cls_pbt_selector}.each_epochs": 1,
    "{cls_pbt_selector}.grace_epochs": 0,
    "{cls_pbt_selector}.save_clone": True,
    "{cls_pbt_selector}.elitist": False,
    "{cls_pbt_selector}.replace_worst": 0.5,
    "{cls_pbt_selector}.copy_best": 0.5,

    "cls_pbt_targets": "OptimizationTarget, OptimizationTarget",
    "{cls_pbt_targets#0}.key": "val/loss",
    "{cls_pbt_targets#0}.maximize": False,
    "{cls_pbt_targets#1}.key": "val/clones/EMAClone/0.999/loss",
    "{cls_pbt_targets#1}.maximize": False,

    "cls_pbt_mutations": "OptimizerPbtMutation, RegularizerPbtMutation",
    "{cls_pbt_mutations#0}.p": 1.0,
    "{cls_pbt_mutations#0}.init_factor": 2,
    "{cls_pbt_mutations#0}.multiplier_smaller": 0.8,
    "{cls_pbt_mutations#0}.multiplier_larger": 1.2,
    "{cls_pbt_mutations#0}.optimizer_index": 0,
    "{cls_pbt_mutations#1}.p": 1.0,
    "{cls_pbt_mutations#1}.init_factor": 1.2,
    "{cls_pbt_mutations#1}.multiplier_smaller": 0.8,
    "{cls_pbt_mutations#1}.multiplier_larger": 1.2,
    "{cls_pbt_mutations#1}.regularizer_name": "DropOutRegularizer",
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
