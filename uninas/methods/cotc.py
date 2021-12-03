import os
import torch
import torch.nn.functional as F
from uninas.models.networks.uninas.search import SearchUninasNetwork
from uninas.data.abstract import AbstractBatchAugmentation
from uninas.methods.abstract_method import AbstractOptimizationMethod
from uninas.methods.strategy_manager import StrategyManager
from uninas.methods.random import RandomChoiceStrategy
from uninas.optimization.estimators.abstract import AbstractEstimator
from uninas.optimization.cream.matching_board import PrioritizedMatchingBoard
from uninas.training.optimizers.abstract import WrappedOptimizer
from uninas.training.result import ResultValue, LogResult
from uninas.utils.args import MetaArgument, Argument, Namespace
from uninas.utils.loggers.python import log_headline, log_in_columns, LoggerManager
from uninas.register import Register


@Register.method(search=True)
class CreamOfTheCropMethod(AbstractOptimizationMethod):
    """
    Randomly sample 1 out of the available options and a similar teacher from the prioritized paths,
    keep track of well performing prioritized paths (assumed pareto optimal with respect to the targets)

    Cream of the Crop: Distilling Prioritized Paths For One-Shot Neural Architecture Search
    https://arxiv.org/pdf/2010.15821.pdf
    https://github.com/microsoft/cream
    """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        assert isinstance(self.net, SearchUninasNetwork)

        # disable random architecture changes for each forward pass
        self.net.set_forward_strategy(forward_strategy=False)

        # targets to optimize the pareto front for
        self.targets = []
        for i, target_cls in enumerate(self._parsed_meta_arguments(Register.optimization_targets, 'cls_cotc_targets', hparams, index=None)):
            target = target_cls.from_args(hparams, index=i)
            key = target.get_key()
            assert key.startswith('train'), 'target key "%s" is not available during training' % key
            self.targets.append(target)

        # estimators that can provide some metrics for those targets
        self.estimators = []
        for i, estimator_cls in enumerate(self._parsed_meta_arguments(Register.hpo_estimators, 'cls_arc_estimators', hparams, index=None)):
            assert issubclass(estimator_cls, AbstractEstimator)
            estimator = estimator_cls(hparams, index=i)
            self.estimators.append(estimator)

        # optional specific meta matching network
        matching_net = None
        for net_cls in self._parsed_meta_arguments(Register.networks, 'cls_mmn_network', hparams, index=None):
            matching_net = net_cls.from_args(hparams, index=None)
            break

        # pareto front board
        board_size, grace_epochs = self._parsed_arguments(['board_size', 'grace_epochs'], hparams, index=None)
        match_batch_size, mmn_batch_size, mmn_batch_average =\
            self._parsed_arguments(['match_batch_size', 'mmn_batch_size', 'mmn_batch_average'], hparams, index=None)
        sel_t, sel_ui = self._parsed_arguments(['select_teacher', 'select_update_iter'], hparams, index=None)
        self.board = PrioritizedMatchingBoard(
            board_size, grace_epochs, sel_t, sel_ui, self.data_set.get_label_shape(), match_batch_size,
            average_mmn_batches=mmn_batch_average, mmn_batch_size=mmn_batch_size, matching_net=matching_net)

        # meta matching network and search network optimizers
        self._optimizer_net = None
        self._optimizer_mmn = None
        self._current_lr_ratio = None

    def on_epoch_start(self, *args, **kwargs) -> dict:
        """
        when the trainer starts a new epoch
        if the method stops early, the is_last flag will never be True
        """
        # set the meta matching network and search network optimizers here (doing it in __init__ breaks DDP)
        # remove the weight decay for the MMN
        if self._optimizer_net is None:
            optimizers, _ = self.configure_optimizers()
            self._optimizer_net = optimizers[0]
            self._optimizer_mmn = self._cls_optimizers[0].from_args(
                self.hparams, 0, self.amp_scaler, named_params=self.board.named_parameters(),
                kwargs_changes=dict(weight_decay=0.0))
            self._optimizer_mmn.set_optimizer_lr(self._parsed_argument('mmn_lr', self.hparams, index=None))
            self._current_lr_ratio = None

        return super().on_epoch_start(*args, **kwargs)

    @classmethod
    def meta_args_to_add(cls, num_optimizers=1, search=True) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        estimators = Register.hpo_estimators.filter_match_all(requires_trainer=False, requires_method=False,
                                                              requires_bench=False)
        return super().meta_args_to_add() + [
            MetaArgument('cls_cotc_targets', Register.optimization_targets, allowed_num=(1, -1), allow_duplicates=True,
                         help_name='targets for path selection, the keys must be in the log_dict'),
            MetaArgument('cls_arc_estimators', estimators, allowed_num=(0, -1), allow_duplicates=True,
                         help_name='estimators to enable optimizing further targets, available as estimators/{name}'),
            MetaArgument('cls_mmn_network', Register.networks, allowed_num=(0, 1), use_index=False,
                         help_name='optional meta matching network'),
        ]

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('board_size', default=10, type=int, help='number of prioritized architectures'),
            Argument('pre_prob', default="", type=str, help='[float] prior probabilities for the path options'),
            Argument('grace_epochs', default=0, type=int, help='epochs before prioritizing paths'),
            Argument('select_teacher', default='meta', type=str, choices=['meta', 'value1', 'l1', 'random'], help='how to select a teacher network'),
            Argument('select_update_iter', default=200, type=int, help='update the matching network each n batches'),
            Argument('match_batch_size', default=8, type=int, help='mini-batch size for student-teacher matching'),
            Argument('mmn_lr', default=1e-4, type=float, help='learning rate for the meta matching network'),
            Argument('mmn_batch_size', default=-1, type=int, help='mini-batch size for the MMN updates, full if <0'),
            Argument('mmn_batch_average', default="True", type=str, is_bool=True, help='average MMN outputs for teacher selection, else concat inputs'),
        ]

    def setup_strategy(self) -> StrategyManager:
        """ set up the strategy for architecture weights """
        pre_prob = self._parsed_argument('pre_prob', self.hparams, index=None, split_=float)
        return StrategyManager().add_strategy(RandomChoiceStrategy(self.max_epochs, fixed_probabilities=pre_prob))

    def _generic_step(self, batch, batch_idx: int, key: str, batch_augments: AbstractBatchAugmentation,
                      only_losses=False, **net_kwargs) -> (torch.Tensor, {str: torch.Tensor}):
        """
        generic forward pass
        sample architectures and keep the board updated

        :param batch: (inputs, outputs)
        :param batch_idx:
        :param key: train/val/test
        :param batch_augments:
        :param only_losses: do not compute metrics or give the architecture strategies feedback
                            ignored here, as the metrics are required for network ranking
        :param net_kwargs:
        :return: network output, log dict
        """

        if key == self._key_train:
            log_dct = {}
            inputs, targets = self._generic_data(batch, batch_augments)

            # sample architecture of the training forward pass
            self.net.forward_strategy()
            arc = self.strategy_manager.get_all_finalized_indices(unique=True, flat=True)

            # use profilers to add usable info
            estimator_constraints = []
            for estimator in self.estimators:
                v = estimator.evaluate_tuple(arc)
                estimator_constraints.append(estimator.is_in_constraints(v))
                log_dct['%s/estimators/%s' % (key, estimator.key)] =\
                    ResultValue(torch.tensor([v], device=inputs.device), inputs.shape[0])

            # forward pass meta matching network and search network
            self.board.update_matching(inputs, targets, arc, self.net, self._optimizer_net, self._optimizer_mmn,
                                       self._loss, self.current_epoch, batch_idx, self._current_lr_ratio)
            with self.amp_autocast:
                logits = self(inputs, **net_kwargs)
            with torch.no_grad():
                for metric in self.metrics:
                    log_dct.update(metric.evaluate(self.net, inputs, logits, targets, key))

            # how well the arc does on the target metrics (smaller is better)
            log_dct_values, _ = LogResult.split_log_dict(log_dct)
            target_values = [target.sort_value(log_dict=log_dct_values) for target in self.targets]

            # loss, either directly, or including a teacher
            teacher_logits = None
            if self.board.get_pareto_best().is_empty() or self.board.is_in_grace_time(self.current_epoch):
                loss = self._loss(logits, targets)
                log_dct.update({key + '/loss': ResultValue(loss, inputs.shape[0])})
            else:
                loss_student = self._loss(logits, targets)
                matching_value, teacher_arc = self.board.select_teacher(self.net, arc)

                # soft loss student to teacher
                with torch.no_grad():
                    self.net.forward_strategy(fixed_arc=teacher_arc)
                    teacher_logits = self.net(inputs)
                    soft_target = F.softmax(teacher_logits[-1].detach(), dim=1)

                loss_to_teacher = self._loss(logits, soft_target)
                loss = (matching_value * loss_to_teacher + (2 - matching_value) * loss_student) / 2
                # as implemented in the original. weighted in favor of loss_student, as matching_value is in [0, 1]

                # care for optional other losses
                losses = self.strategy_manager.get_losses(clear=True)
                for k, loss_ in losses.items():
                    log_dct['%s/loss/%s' % (key, k)] = ResultValue(loss_.clone().detach(), inputs.size(0))
                    loss = loss + loss_

                log_dct.update({
                    key + '/loss': ResultValue(loss, inputs.shape[0]),
                    key + '/loss/student': ResultValue(loss_student, inputs.shape[0]),
                    key + '/loss/to_teacher': ResultValue(loss_to_teacher, inputs.shape[0])
                })

            # update board ranking, if the target is in constraints
            if all(estimator_constraints):
                self.board.update_board(self.current_epoch, arc, target_values, inputs, logits, teacher_logits)

            self.strategy_manager.feedback(key, log_dct, self.current_epoch, batch_idx)
            return self.amp_scaler.scale(loss), log_dct

        return super()._generic_step(batch, batch_idx, key, batch_augments, **net_kwargs)

    def optimizer_step(self, *args, optimizer: WrappedOptimizer = None, **kwargs):
        super().optimizer_step(*args, optimizer=optimizer, **kwargs)
        self._current_lr_ratio = optimizer.get_lr_ratio()

    def save_configs(self, cfg_dir: str):
        os.makedirs(cfg_dir, exist_ok=True)
        Register.builder.save_config(self.net.config(finalize=False), cfg_dir, 'search')
        lines = [[t.get_key() for t in self.targets] + ["Architecture"]]
        for entry in self.board.get_pareto_best().get_entries():
            self.net.forward_strategy(fixed_arc=entry.arc)
            Register.builder.save_config(self.net.config(finalize=True), cfg_dir,
                                         'candidate-%s' % '-'.join([str(s) for s in entry.arc]))
            lines.append([t.signed_value(v) for t, v in zip(self.targets, entry.values)] + [entry.arc])

        log_headline(LoggerManager().get_logger(), 'Prioritized paths')
        log_in_columns(LoggerManager().get_logger(), lines, add_bullets=True, num_headers=1)
