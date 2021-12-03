from uninas.main import Main

"""
training a super net

beware that we are using fake data and a test run (only 10 steps/epoch)
"""

config_files = "{path_conf_tasks}/s1_random.run_config, {path_conf_net_search}/fairnas.run_config"


args = {
    "{cls_task}.is_test_run": True,
    "{cls_task}.save_dir": "{path_tmp}/s1/",
    "{cls_task}.save_del_old": True,

    "{cls_trainer}.max_epochs": 6,

    "{cls_data}.fake": True,
    "{cls_data}.dir": "{path_data}/ImageNet_ILSVRC2012/",
    "{cls_data}.batch_size_train": 4,
    "{cls_data}.batch_size_test": -1,
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task(config_files, args_changes=args)
    # print(task.get_method().get_network())
    task.load()
    task.run()
