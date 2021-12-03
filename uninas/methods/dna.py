"""
Blockwisely Supervised Neural Architecture Search with Knowledge Distillation
https://arxiv.org/abs/1911.13053
https://github.com/changlin31/DNA

simplified
 - there are no parallel cells with different lengths, linear transformers should be used instead (SCARLET-NAS)
 - cheaper, each forward pass can train all blocks at once
"""

from typing import Callable, Optional
import torch
import torch.nn as nn
from uninas.models.networks.abstract import AbstractNetwork
from uninas.models.networks.uninas.search import SearchUninasNetwork
from uninas.methods.abstract_method import AbstractOptimizationMethod
from uninas.methods.strategy_manager import StrategyManager
from uninas.training.criteria.abstract import MultiCriterion
from uninas.training.optimizers.abstract import MultiWrappedOptimizer
from uninas.utils.shape import Shape, ShapeList
from uninas.utils.args import MetaArgument, Argument, Namespace
from uninas.utils.loggers.python import LoggerManager, log_headline, log_in_columns
from uninas.register import Register


class SyncBlock:
    def __init__(self):
        self.student_indices = []
        self.teacher_indices = []

    @property
    def first_student_index(self) -> int:
        return self.student_indices[0]

    @property
    def last_student_index(self) -> int:
        return self.student_indices[-1]

    @property
    def first_teacher_index(self) -> int:
        return self.teacher_indices[0]

    @property
    def last_teacher_index(self) -> int:
        return self.teacher_indices[-1]

    @property
    def name(self) -> str:
        return 'sb-%d' % self.first_teacher_index

    def __str__(self) -> str:
        return "%s(%s, student=%s, teacher=%s)" %\
               (self.__class__.__name__, self.name, self.student_indices, self.teacher_indices)


class SyncBlocks:
    def __init__(self, cells_first: int, cells_last: int):
        self.cells_first = cells_first
        self.cells_last = cells_last
        self.used_blocks = list(range(cells_first, cells_last + 1))
        self.blocks = []

    def weight_strategies(self) -> {str: list}:
        return {sb.name: sb.student_indices for sb in self.blocks}

    def align_spatial_shapes(self, teacher_shapes_out: ShapeList, student_shapes_out: ShapeList, split_by_features=False):
        """
        automatically align student to teacher blocks by spatial shapes
        """
        si, blocks, s_out_shape_prev, t_out_shape_prev, c_prev = 0, [], Shape([-1, -1, -1]), Shape([-1, -1, -1]), 0
        for ti in self.used_blocks:
            # start next sync block?
            ts = teacher_shapes_out[ti]
            if (not Shape.same_spatial_sizes(t_out_shape_prev, ts)) or\
                    (split_by_features and (t_out_shape_prev.num_features() != ts.num_features())):
                blocks.append(SyncBlock())
                s_out_shape_prev = student_shapes_out[si]
            t_out_shape_prev = ts
            blocks[-1].teacher_indices.append(ti)

            # find all student blocks that match the spatial size
            for si in range(si, len(student_shapes_out)):
                if split_by_features and (not s_out_shape_prev.num_features() == student_shapes_out[si].num_features()):
                    break
                elif Shape.same_spatial_sizes(ts, student_shapes_out[si]) and (si not in blocks[-1].student_indices):
                    blocks[-1].student_indices.append(si)
                    s_out_shape_prev = student_shapes_out[si]
                else:
                    break

        # ensure that all blocks are taken care of
        self.blocks = [sb for sb in blocks if len(sb.student_indices) > 0 or len(sb.teacher_indices) > 0]
        for sb in self.blocks:
            if len(sb.student_indices) == 0 or len(sb.teacher_indices) == 0:
                raise ValueError("issue with sync block: %s" % sb)


@Register.method(search=True, single_path=True, distill=True)
class DnaMethod(AbstractOptimizationMethod):
    """
    Uses a teacher model to search smaller parts of the network, which imitate the teacher's outputs at certain cells.
    Each block uses its own learning rate and has a multiplier on the loss

    Blockwisely Supervised Neural Architecture Search with Knowledge Distillation
    https://arxiv.org/abs/1911.13053
    https://github.com/changlin31/DNA

    differences to the original:
    - here only one path exists per stage, not multiple in parallel (can use e.g. skip / linear transformers)
    - here a pareto front for each stage is built, and their combinations are searched later for a global pareto front
    """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        self.net.use_forward_mode(mode='cells')
        self.teacher_net.use_forward_mode(mode='cells')
        self.use_forward_mode('custom')

    @classmethod
    def meta_args_to_add(cls, **__) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        # internal networks have many more cells, so a 1-N matching (teacher-student) approach does no longer work
        teachers = Register.networks.filter_match_all(search=False)
        strategies = Register.strategies.filter_match_all(can_hpo=True)

        return super().meta_args_to_add(**__) + [
            MetaArgument('cls_strategy', strategies, help_name='strategy for training', allowed_num=1),
            MetaArgument('cls_teacher_network', teachers, help_name='teacher network', allowed_num=1),
        ]

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('teacher_cells_first', default=1, type=int, help='first teacher cell to distill'),
            Argument('teacher_cells_last', default=6, type=int, help='last teacher cell to distill'),
            Argument('teacher_assert_trained', default="True", type=str, is_bool=True,
                     help='make sure that the teacher model is trained (loaded weights)'),
            Argument('teacher_adapt', default="False", type=str, is_bool=True,
                     help='adapt the teacher model during training (e.g. batchnorm stats)'),
            Argument('split_by_features', default="True", type=str, is_bool=True,
                     help='whether to split a stage if the number of output features changes'),
            Argument('loss_weights', default="", type=str,
                     help='list of floats, multiply the loss of the blocks accordingly. ones if empty, const if scalar.'),
            Argument('optimizer_lr_multipliers', default="", type=str,
                     help='list of floats, multiply the learning rate for each block. ones if empty, const if scalar.'),
        ]

    def train(self, mode: bool = True):
        # make sure the teacher remains in eval mode if it is not allowed to adapt
        s = super().train(mode)
        if not self.teacher_adapt:
            self.teacher_net.eval()
        return s

    def forward_custom(self, x: torch.Tensor, start_cell=-1, end_cell=None) -> [[(torch.Tensor, torch.Tensor)]]:
        """
        x is the data set input, no matter of start_cell
        start_cell/end_cell with respect to the teacher network

        returns nested list: [block_idx, (teacher output, net output)]
        """
        if start_cell >= 0:
            with torch.no_grad():
                x = self.teacher_net(x, start_cell=-1, end_cell=start_cell-1)[0]
        outputs = []

        for i, sb in enumerate(self.sync_blocks.blocks):
            if sb.last_teacher_index < start_cell:
                outputs.append([])
                continue

            if start_cell == -1:
                y = self.net(x, start_cell=-1, end_cell=sb.last_student_index)
            else:
                scaled_x = self.channel_scale_in[i](x)
                y = self.net(scaled_x, start_cell=sb.first_student_index, end_cell=sb.last_student_index)

            with torch.no_grad():
                x = self.teacher_net(x, start_cell=start_cell, end_cell=sb.last_teacher_index)[0]

            outputs.append([(x, self.channel_scale_out[i](y_)) for y_ in y])

            start_cell = sb.last_teacher_index + 1
            if sb.last_teacher_index == end_cell:
                break

        return outputs

    def setup_strategy(self) -> StrategyManager:
        """
        set up strategy for architecture weights, the teacher network and teacher-student alignment
        """
        logger = LoggerManager().get_logger()
        strategy_cls = self._parsed_meta_argument(Register.strategies, 'cls_strategy', self.hparams, index=None)
        cells_first, cells_last = self._parsed_arguments(['teacher_cells_first', 'teacher_cells_last'], self.hparams)
        split_by_features = self._parsed_argument('split_by_features', self.hparams)
        self.sync_blocks = SyncBlocks(cells_first, cells_last)
        probing_strategy_name = '__probing__'
        sm = StrategyManager()

        # set the teacher network
        log_headline(logger, 'Teacher Model:', target_len=80)
        cls_teacher_network = self._parsed_meta_argument(Register.networks, 'cls_teacher_network', self.hparams, index=None)
        teacher_assert_trained = self._parsed_argument('teacher_assert_trained', self.hparams, index=None)
        self.teacher_adapt = self._parsed_argument('teacher_adapt', self.hparams, index=None)
        self.teacher_net = cls_teacher_network.from_args(self.hparams)
        self.teacher_net.build(s_in=self.data_set.get_data_shape(), s_out=self.data_set.get_label_shape())
        assert isinstance(self.teacher_net, AbstractNetwork)
        if teacher_assert_trained:
            assert self.teacher_net.has_loaded_weights(), "The teacher network did not load weights!"
        if not self.teacher_adapt:
            self.teacher_net.disable_state_dict()  # prevent the teacher net from saving/loading weights
        self.teacher_net.eval()
        teacher_shapes_in = self.teacher_net.get_cell_input_shapes(flatten=True)
        teacher_shapes_out = self.teacher_net.get_cell_output_shapes(flatten=True)

        # align teacher and student networks
        sm.add_strategy(strategy_cls(self.max_epochs, name=probing_strategy_name))

        log_headline(logger, "Probing network to align shapes", target_len=80)
        net_cls = self._parsed_meta_argument(Register.networks, 'cls_network', self.hparams, index=None)
        probe_net = net_cls.from_args(self.hparams, weight_strategies=probing_strategy_name)
        probe_net.build(s_in=self.data_set.get_data_shape(), s_out=self.data_set.get_label_shape())
        assert isinstance(probe_net, SearchUninasNetwork)

        student_shapes_in = probe_net.get_cell_input_shapes(flatten=True)
        student_shapes_out = probe_net.get_cell_output_shapes(flatten=True)
        self.sync_blocks.align_spatial_shapes(teacher_shapes_out, student_shapes_out,
                                              split_by_features=split_by_features)

        del probe_net
        sm.delete_strategy(probing_strategy_name)

        # create the required amount of weight strategies
        for k in self.sync_blocks.weight_strategies().keys():
            sm.add_strategy(strategy_cls(self.max_epochs, name=k))

        # create the required channel scale operations
        log_headline(logger, 'Student-teacher alignment:', target_len=80)
        logger.info('Scaling to get the correct number of features:')
        self.channel_scale_in = nn.ModuleList()
        self.channel_scale_out = nn.ModuleList()
        rows = [("name", "student", "teacher", "scale in", "scale out",
                 "teacher in", "", "student in", "student out", "", "teacher out")]
        for sb in self.sync_blocks.blocks:
            ts_in, ts_out = teacher_shapes_in[sb.first_teacher_index], teacher_shapes_out[sb.last_teacher_index]
            str_in, str_out = '', ''
            # scale block input
            ss_in = student_shapes_in[sb.first_student_index]
            if ts_in.num_features() == ss_in.num_features():
                self.channel_scale_in.append(nn.Identity())
            else:
                c1_in, c2_in = ts_in.num_features(), ss_in.num_features()
                self.channel_scale_in.append(nn.Conv2d(c1_in, c2_in, kernel_size=1, bias=False))
                str_in = '(%d -> %d)' % (c1_in, c2_in)
            # scale block output
            ss_out = student_shapes_out[sb.last_student_index]
            if ss_out == ts_out:
                self.channel_scale_out.append(nn.Identity())
            else:
                c1_out, c2_out = ss_out.num_features(), ts_out.num_features()
                self.channel_scale_out.append(nn.Conv2d(c1_out, c2_out, kernel_size=1, bias=False))
                str_out = '(%d -> %d)' % (c1_out, c2_out)
            rows.append((sb.name, str(sb.student_indices), str(sb.teacher_indices), str_in, str_out,
                         ts_in.str(), "-->", ss_in.str(), ss_out.str(), "-->", ts_out.str()))
        log_in_columns(logger, rows, add_bullets=True, num_headers=1)

        return sm

    def _get_new_network(self) -> SearchUninasNetwork:
        log_headline(LoggerManager().get_logger(), "Real network", target_len=80)
        net_cls = self._parsed_meta_argument(Register.networks, 'cls_network', self.hparams, index=None)
        net = net_cls.from_args(self.hparams, weight_strategies=self.sync_blocks.weight_strategies())
        assert isinstance(net, SearchUninasNetwork)
        return net

    def get_weights_criterion(self) -> (list, MultiCriterion):
        cells_first, cells_last = self._parsed_arguments(['teacher_cells_first', 'teacher_cells_last'], self.hparams)
        n = cells_last - cells_first + 1
        weights = self._parsed_argument('loss_weights', self.hparams, split_=float)
        if len(weights) == 0:
            weights = [1]*n
        elif len(weights) == 1:
            weights = [weights[0]]*n
        assert len(weights) == n, "number of weights (%d) and cells (%d) must match" % (len(weights), n)
        cls_criterion = self._parsed_meta_argument(Register.criteria, 'cls_criterion', self.hparams, index=None)
        criterion = MultiCriterion.from_args(weights, cls_criterion, self.data_set, self.hparams)
        LoggerManager().get_logger().info("Weighting model block loss: %s" % str(weights))
        return weights, criterion

    def optimizer_step(self, *args, epoch: int = None, optimizer: MultiWrappedOptimizer = None,
                       optimizer_closure: Optional[Callable] = None, **kwargs):
        optimizer.step_all(closure=optimizer_closure)
        self.amp_scaler.update()
        optimizer.zero_grad_all()

    def configure_optimizers(self) -> (list, list):
        """ get optimizers/schedulers """
        assert len(self._cls_schedulers) <= len(self._cls_optimizers) == 1
        n = len(self.sync_blocks.blocks)
        lrs_mults = self._parsed_argument('optimizer_lr_multipliers', self.hparams, split_=float)
        if len(lrs_mults) == 0:
            lrs_mults = [1]*n
        elif len(lrs_mults) == 1:
            lrs_mults = [lrs_mults[0]]*n
        assert len(lrs_mults) == n, "number of LR multipliers (%d) and cells (%d) must match" % (len(lrs_mults), n)

        # assign parameters to their respective blocks, stem to first, heads to last
        params = [[] for _ in range(n)]
        params[0].extend(list(self.net.get_network().get_stem().named_parameters()))
        params[-1].extend(list(self.net.get_network().get_heads().named_parameters()))
        cells = self.net.get_network().get_cells()

        for i, sb in enumerate(self.sync_blocks.blocks):
            for idx in sb.student_indices:
                params[i].extend((cells[idx].named_parameters()))
            params[i].extend(self.channel_scale_in[i].named_parameters())
            params[i].extend(self.channel_scale_out[i].named_parameters())

        # one optimizer instance per block
        optimizers, schedulers = [], []
        for i, lr in enumerate(lrs_mults):
            optimizer = self._cls_optimizers[0].from_args(self.hparams, index=0, scaler=self.amp_scaler, named_params=params[i])
            optimizer.set_optimizer_lr(lr, is_multiplier=True)
            optimizers.append(optimizer)
            if len(self._cls_schedulers) > 0:
                scheduler = self._cls_schedulers[0].from_args(self.hparams, optimizer, self.max_epochs, index=0)
                if scheduler is not None:
                    schedulers.append(scheduler)
        return [MultiWrappedOptimizer(optimizers)], schedulers
