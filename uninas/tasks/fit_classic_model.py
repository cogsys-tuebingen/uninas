import time
from typing import Union
import numpy as np
import torch
from uninas.tasks.abstract import AbstractTask
from uninas.utils.args import MetaArgument, Argument, Namespace
from uninas.utils.loggers.python import log_headline, log_in_columns
from uninas.utils.loggers.exp import AbstractExpLogger
from uninas.register import Register


@Register.task(search=True)
class FitClassicModelTask(AbstractTask):
    """
    Fit a classic ML model (not a network) to some data
    """

    def __init__(self, args: Namespace, *args_, **kwargs):
        super().__init__(args, *args_, **kwargs)
        log_headline(self.logger, "Setting up")

        # model
        cls_model = self._parsed_meta_argument(Register.models, 'cls_model', args, index=None)
        self.model = cls_model.from_args(args, index=None)
        self._fit_loaded_model = self._parsed_argument('fit_loaded_model', args, index=None)
        self._model_is_loaded = False

        # data
        data_set_cls = self._parsed_meta_argument(Register.data_sets, 'cls_data', args, index=None)
        self.data_set = data_set_cls.from_args(args, index=None)
        self._undo_label_normalization = self._parsed_argument('undo_label_normalization', args, index=None)

        # reduce the data set size to fit significantly in test runs
        max_num = 100 if self.is_test_run else -1
        self.data_train = self.data_set.get_full_train_data(to_numpy=True, num=max_num)
        self.data_valid = self.data_set.get_full_valid_data(to_numpy=True, num=max_num)
        self.data_test = self.data_set.get_full_test_data(to_numpy=True, num=max_num)

        # exp logger
        logger_save_dir = '%sexp/' % self.save_dir
        self.exp_logger = AbstractExpLogger.collection(logger_save_dir, args, self._parsed_meta_arguments(Register.exp_loggers, 'cls_exp_loggers', args, index=None))
        self.exp_logger.log_hyperparams(args)

        # metrics
        cls_metrics = self._parsed_meta_arguments(Register.metrics, 'cls_metrics', args, index=None)
        self.metrics = [m.from_args(args, i, self.data_set, [1.0]) for i, m in enumerate(cls_metrics)]
        for m in self.metrics:
            m.on_epoch_start(0, is_last=True)

        # log
        rows = [
            ("Data set", self.data_set.str()),
            (" > train", self.data_set.list_train_transforms()),
            (" > valid", self.data_set.list_valid_transforms()),
            (" > test", self.data_set.list_test_transforms()),
        ]
        if max_num > 0:
            rows.append((" > LIMIT", "only %d data points used for train/valid/test each" % max_num))
        rows.append(("Model", self.model.str()))
        if len(self.metrics) > 0:
            rows.append(("Metrics", ""))
            for i, x in enumerate(self.metrics):
                rows.append((" (%d)" % i, x.str()))
        log_in_columns(self.logger, rows)

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        models = Register.models.filter_match_all(can_fit=True)
        return super().meta_args_to_add() + [
            MetaArgument('cls_data', Register.data_sets, help_name='data set', allowed_num=1),
            MetaArgument('cls_model', models, help_name='model to fit', allowed_num=1),
            MetaArgument('cls_exp_loggers', Register.exp_loggers, help_name='experiment logger', allow_duplicates=True),
            MetaArgument('cls_metrics', Register.metrics, help_name='metrics to evaluate', allow_duplicates=True),
        ]

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('fit_loaded_model', default="False", type=str, help='fit a model again if it was loaded successfully', is_bool=True),
            Argument('undo_label_normalization', default="False", type=str, help='undo normalization for metrics', is_bool=True),
        ]

    @classmethod
    def _model_save_path(cls, dir_: str) -> str:
        return '%s/model.pt' % dir_

    def _compute_metrics(self, inputs: Union[np.array, None], targets: Union[np.array, None], key: str) -> dict:
        """ compute metrics on non-normalized labels """
        log_dict = {}
        if (inputs is not None) and (targets is not None):
            # predict
            predictions = self.model.predict(inputs)
            if len(predictions.shape) < len(targets.shape):
                predictions = np.expand_dims(predictions, axis=-1)

            # undo normalization?
            if self._undo_label_normalization:
                targets = self.data_set.undo_label_normalization(targets)
                predictions = self.data_set.undo_label_normalization(predictions)

            # need tensors
            inputs_ = torch.Tensor(inputs)
            targets = torch.Tensor(targets)
            predictions = torch.Tensor(predictions)

            # compute metrics
            for m in self.metrics:
                log_dict.update(m.evaluate(None, inputs_, [predictions], targets, key))
                log_dict.update(m.eval_accumulated_stats(save_dir="%s/metrics/%s/" % (self.save_dir, m.get_log_name()), key=key))

        return log_dict

    def _load(self, checkpoint_dir: str) -> bool:
        """ load """
        path = self._model_save_path(checkpoint_dir)
        loaded = self.model.load(path)
        if loaded:
            self.logger.info("Successfully loaded the model from %s" % path)
            self._model_is_loaded = True
        return loaded

    def _run(self):
        log_dict = {}

        if self.data_train[0] is not None:
            inputs, targets = self.data_train
            log_headline(self.logger, "Fitting model")
            if self._fit_loaded_model or (not self._model_is_loaded):
                self.logger.info("Training data shape: %s" % repr(inputs.shape))
                t0 = time.time()
                self.model.fit(inputs, targets)
                td = time.time() - t0
                log_dict['time'] = torch.Tensor([td])
                self.logger.info("Fitting done after %.2f seconds" % td)
            else:
                self.logger.info("Skipped, already loaded a model, re-fitting is disabled")

            path = self._model_save_path(self.save_dir)
            self.model.save(path)
            self.logger.info("Saved model to %s" % path)

        log_headline(self.logger, "Evaluating")
        log_dict.update(self._compute_metrics(*self.data_train, "train"))
        log_dict.update(self._compute_metrics(*self.data_valid, "valid"))
        log_dict.update(self._compute_metrics(*self.data_test, "test"))

        rows = [(k, v.item()) for k, v in log_dict.items()]
        log_in_columns(self.logger, rows, add_bullets=True)
        self.exp_logger.log_metrics({k: v.item() for k, v in log_dict.items()})
