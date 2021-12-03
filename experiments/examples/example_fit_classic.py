from uninas.main import Main

"""
fit a classic ML model (e.g. linear, SVM, random forest regressor) to profiling data
"""


args = {
    "cls_task": "FitClassicModelTask",
    "{cls_task}.save_dir": "{path_tmp}/profiled/fit/",
    "{cls_task}.save_del_old": True,
    "{cls_task}.is_test_run": False,
    "{cls_task}.fit_loaded_model": True,
    "{cls_task}.undo_label_normalization": False,

    "cls_exp_loggers": "TensorBoardExpLogger",

    "cls_data": "ProfiledData",
    "{cls_data}.valid_split": 0.0,

    # "{cls_data}.dir": "{path_data}/profiling/cpu_30k/",
    # "{cls_data}.file_name": "mini.pt",
    # "{cls_data}.file_name": "data_standalone.pt",
    # "{cls_data}.file_name": "data_overcomplete.pt",

    "{cls_data}.dir": "{path_data}/profiling/HW-NAS/",
    "{cls_data}.file_name": "cifar10-edgegpu_energy.pt",
    # "{cls_data}.file_name": "ImageNet16-120-raspi4_latency.pt",

    "{cls_data}.cast_one_hot": True,
    "{cls_data}.normalize_labels": True,
    "{cls_data}.train_num": 500,  # -1

    "cls_augmentations": "",

    # "cls_model": "LinearRegressionSklearnModel",

    # "cls_model": "SVMRegressionSklearnModel",
    # "{cls_model}.kernel": "rbf",

    # "cls_model": "RandomForestRegressorSklearnModel",
    # "{cls_model}.n_estimators": 50,
    # "{cls_model}.criterion": "mse",
    # "{cls_model}.max_samples": 2000,

    "cls_model": "RegressionXGBoostModel",
    "{cls_model}.n_estimators": 50,
    "{cls_model}.objective": "reg:squarederror",  # reg:linear, reg:squarederror

    "cls_metrics": "CriterionMetric, CriterionMetric, CorrelationsMetric, FitDistributionsMetric",
    "{cls_metrics#0}.criterion": 'L1Criterion',
    "{cls_metrics#1}.criterion": 'L2Criterion',
    "{cls_metrics#2}.correlations": 'KendallTauNasMetric, SpearmanNasMetric',
    "{cls_metrics#3}.fit_normal": True,
    "{cls_metrics#3}.test_ks": True,
}

if __name__ == "__main__":
    # ignore the command line, use "args" instead
    task = Main.new_task([], args_changes=args)
    task.load()
    task.run()
