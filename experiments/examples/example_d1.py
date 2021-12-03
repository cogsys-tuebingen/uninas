from uninas.main import Main

"""
search the architecture of a small network via DARTS algorithm

beware that we are using fake data
"""


args = {
    "cls_task": "SingleSearchTask",
    "{cls_task}.save_dir": "{path_tmp}/d1/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": True,

    "cls_device": "CudaDevicesManager",
    "{cls_device}.num_devices": 1,
    
    "cls_trainer": "SimpleDDPTrainer",   # SimpleTrainer, SimpleDDPTrainer
    "{cls_trainer}.max_epochs": 3,
    "{cls_trainer}.eval_last": 2,
    "{cls_trainer}.test_last": 2,
    "{cls_trainer}.sharing_strategy": 'file_system',

    "cls_exp_loggers": "TensorBoardExpLogger",
    "{cls_exp_loggers#0}.log_graph": False,

    "cls_callbacks": "CheckpointCallback",
    "{cls_callbacks#0}.save_clone": True,
    "{cls_callbacks#0}.top_n": 0,
    "{cls_callbacks#0}.key": "train/loss",
    "{cls_callbacks#0}.minimize_key": True,

    "cls_data": "Cinic10Data",
    # "cls_data": "Cifar10Data",
    "{cls_data}.fake": True,
    "{cls_data}.valid_split": 0.0,
    "{cls_data}.batch_size_train": 2,

    "cls_augmentations": "DartsCifarAug",
    
    "cls_method": "DartsSearchMethod",  # DartsSearchMethod, GdasSearchMethod

    "cls_network": "SearchUninasNetwork",

    "cls_network_body": "StackedCellsNetworkBody",
    "{cls_network_body}.cell_order": "n, n, r, n, n, r, n, n",
    "{cls_network_body}.features_first_cell": 64,

    "cls_network_stem": "DartsCifarStem",
    "{cls_network_stem}.features": 48,

    "cls_network_heads": "ClassificationHead",
    "{cls_network_heads#0}.weight": 1.0,
    "{cls_network_heads#0}.cell_idx": -1,
    "{cls_network_heads#0}.persist": "True",

    "cls_network_cells": "DartsCNNSearchCell, DartsCNNSearchCell",
    "{cls_network_cells#0}.name": "n",
    "{cls_network_cells#0}.arc_key": "n",
    "{cls_network_cells#0}.arc_shared": True,
    "{cls_network_cells#0}.features_mult": 1,
    "{cls_network_cells#0}.stride": 1,
    "{cls_network_cells#0}.num_concat": 4,
    "{cls_network_cells#0}.num_blocks": 4,
    "{cls_network_cells#0}.cls_block": "DartsCNNSearchBlock",
    "{cls_network_cells#1}.name": "r",
    "{cls_network_cells#1}.arc_key": "r",
    "{cls_network_cells#1}.arc_shared": True,
    "{cls_network_cells#1}.features_mult": 2,
    "{cls_network_cells#1}.stride": 2,
    "{cls_network_cells#1}.num_concat": 4,
    "{cls_network_cells#1}.num_blocks": 4,
    "{cls_network_cells#1}.cls_block": "DartsCNNSearchBlock",

    "cls_network_cells_primitives": "DartsPrimitives, DartsPrimitives",

    "cls_metrics": "AccuracyMetric",

    "cls_initializers": "",
    
    "cls_regularizers": "DropOutRegularizer, DropPathRegularizer",
    "{cls_regularizers#1}.max_prob": 0.3,
    
    "cls_criterion": "CrossEntropyCriterion",

    "cls_optimizers": "SGDOptimizer, AdamOptimizer",
    "{cls_optimizers#0}.lr": 0.05,
    "{cls_optimizers#0}.momentum": 0.5,
    "{cls_optimizers#1}.lr": 0.03,
    "{cls_optimizers#1}.weight_decay": 1e-2,

    "cls_schedulers": "CosineScheduler, ConstantScheduler",
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
