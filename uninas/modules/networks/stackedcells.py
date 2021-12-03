import torch
import torch.nn as nn
from typing import Union
from collections import OrderedDict
from uninas.methods.strategy_manager import StrategyManagerDefault
from uninas.modules.networks.abstract import AbstractNetworkBody
from uninas.modules.modules.abstract import AbstractModule
from uninas.modules.cells.abstract import AbstractCell, SearchCellInterface, FixedSearchCellInterface
from uninas.modules.primitives.abstract import PrimitiveSet
from uninas.utils.misc import split
from uninas.utils.shape import Shape, ShapeList
from uninas.utils.args import MetaArgument, Argument, Namespace
from uninas.utils.torch.misc import count_parameters
from uninas.utils.loggers.python import log_in_columns, LoggerManager
from uninas.register import Register


@Register.network_body()
class StackedCellsNetworkBody(AbstractNetworkBody):
    """
    A model made from stacking cells
    Purely sequential, one stem with n outputs, every cell has n inputs and n outputs
    """

    def __init__(self, stem: AbstractModule, heads: nn.ModuleList,
                 cell_configs: dict, cell_partials: dict, cell_order: Union[str, list],
                 weight_strategies: Union[dict, str, None] = None, **kwargs_to_save):
        """
        :param stem:
        :param heads:
        :param cell_configs: {name: config} dict of saved cells
        :param cell_partials: {name: Callable(cell_index)} to create new (search) cells
        :param cell_order: str or [str], will be split to [str]
        :param weight_strategies: override default strategies,
                                  {strategy name: [cell indices]}, or name used for all, or None
        :param kwargs_to_save:
        """
        super().__init__(**kwargs_to_save)
        self._add_to_submodules(stem=stem)
        self._add_to_submodule_lists(heads=heads)
        self._add_to_kwargs_np(cell_configs=cell_configs, cell_partials={})
        if isinstance(cell_order, str):
            cell_order = split(cell_order)
        self._add_to_kwargs(cell_order=cell_order)
        self._add_to_print_kwargs(weight_strategies=weight_strategies)
        self._cell_partials = cell_partials
        self.cells = nn.ModuleList()
        self._head_positions = {}
        self._update_heads()

    def get_stem(self) -> AbstractModule:
        return self.stem

    def get_cells(self) -> nn.ModuleList:
        return self.cells

    def get_heads(self) -> nn.ModuleList:
        return self.heads

    def get_last_head(self) -> nn.Module:
        return self._head_positions[len(self.cells)-1]

    def _update_heads(self):
        heads = [(len(self.cell_order)-1 if h.cell_idx == -1 else h.cell_idx, h) for h in self.heads]
        self._head_positions = OrderedDict(sorted(heads, key=lambda v: v[0]))
        assert len(self._head_positions) == len(self.heads), 'Can not have multiple heads at the same position!'
        assert self._head_positions.get(len(self.cell_order)-1).persist, 'The final head must stay in the config'

    def str(self, *args, **kwargs):
        return super().str(*args, **kwargs, add_sl=dict(Cells=self.cells))

    def set_dropout_rate(self, p=None, stem_p=None, cells_p=None) -> int:
        """ set the dropout rate of every dropout layer to p, no change for p=None, only change the head by default """
        n = self.stem.set_dropout_rate(stem_p)
        for m in self.cells:
            n += m.set_dropout_rate(cells_p)
        for m in self.heads:
            n += m.set_dropout_rate(p)
        return n

    def get_head_weightings(self) -> [float]:
        """ get the weights of all heads, in order (the last head at -1 has to be last) """
        return [v.weight for v in self._head_positions.values()]

    def _build(self, s_in: Shape, s_out: Shape) -> ShapeList:
        LoggerManager().get_logger().info('Building %s:' % self.__class__.__name__)
        rows = [('cell index', 'name', 'class', 'input shapes', '', 'output shapes', '#params')]

        def get_row(idx, name: str, obj: AbstractModule) -> tuple:
            s_in_str = obj.get_shape_in().str()
            s_inner = obj.get_cached('shape_inner')
            s_inner_str = '' if s_inner is None else s_inner.str()
            s_out_str = obj.get_shape_out().str()
            return str(idx), name, obj.__class__.__name__, s_in_str, s_inner_str, s_out_str, count_parameters(obj)

        s_out_data = s_out.copy()
        out_shapes = self.stem.build(s_in)
        final_out_shapes = []
        rows.append(get_row('', '-', self.stem))

        # cells and (aux) heads
        updated_cell_order = []
        for i, cell_name in enumerate(self.cell_order):
            strategy_name, cell = self._get_cell(name=cell_name, cell_index=i)
            assert self.stem.num_outputs() == cell.num_inputs() == cell.num_outputs(), 'Cell does not fit the network!'
            updated_cell_order.append(cell.name)
            s_ins = out_shapes[-cell.num_inputs():]
            with StrategyManagerDefault(strategy_name):
                s_out = cell.build(s_ins.copy(),
                                   features_mul=self.features_mul,
                                   features_fixed=self.features_first_cell if i == 0 else -1)
            out_shapes.extend(s_out)
            rows.append(get_row(i, cell_name, cell))
            self.cells.append(cell)

            # optional (aux) head after every cell
            head = self._head_positions.get(i, None)
            if head is not None:
                if head.weight > 0:
                    final_out_shapes.append(head.build(s_out[-1], s_out_data))
                    rows.append(get_row('', '-', head))
                else:
                    LoggerManager().get_logger().info('not adding head after cell %d, weight <= 0' % i)
                    del self._head_positions[i]
            else:
                assert i != len(self.cell_order) - 1, "Must have a head after the final cell"

        # remove heads that are impossible to add
        for i in self._head_positions.keys():
            if i >= len(self.cells):
                LoggerManager().get_logger().warning('Can not add a head after cell %d which does not exist, deleting the head!' % i)
                head = self._head_positions.get(i)
                for j, head2 in enumerate(self.heads):
                    if head is head2:
                        self.heads.__delitem__(j)
                        break

        s_out = ShapeList(final_out_shapes)
        rows.append(('complete network', '', '', self.get_shape_in().str(), '', s_out.str(), count_parameters(self)))
        log_in_columns(LoggerManager().get_logger(), rows, start_space=4)
        self.set(cell_order=updated_cell_order)
        return s_out

    def _get_cell(self, name: str, cell_index: int) -> (Union[str, None], AbstractCell):
        """ get a cell and the used weight strategy name, either from known config or an already partially built one """
        if name in self._cell_partials:
            strategy_name = None
            if self.weight_strategies is None or isinstance(self.weight_strategies, str):
                strategy_name = self.weight_strategies
            elif isinstance(self.weight_strategies, dict):
                for k, v in self.weight_strategies.items():
                    if cell_index in v:
                        strategy_name = k
                        break
                if strategy_name is None:
                    raise KeyError("missing name of weight strategy at cell index %d" % cell_index)
            else:
                raise NotImplementedError("unknown format for weight strategies: %s" % type(self.weight_strategies))
            with StrategyManagerDefault(strategy_name):
                return strategy_name, self._cell_partials[name](cell_index=cell_index)
        if name in self.cell_configs:
            return None, Register.builder.from_config(self.cell_configs[name])
        raise ModuleNotFoundError('Could not find a cell with name "%s"' % name)

    def forward(self, x: torch.Tensor) -> [torch.Tensor]:
        """
        returns list of all heads' outputs
        the heads are sorted by ascending cell order
        """
        logits_head = []
        x = self.stem(x)
        for i, m in enumerate(self.cells):
            x = m(x)
            head = self._head_positions.get(i)
            if head:
                logits_head.append(head(x[-1]))
        return logits_head

    def specific_forward(self, x: Union[torch.Tensor, list], start_cell=-1, end_cell=None) -> [torch.Tensor]:
        """
        can execute specific part of the network,
        returns result after end_cell
        """

        # stem, -1
        if start_cell <= -1:
            x = self.stem(x)
        if end_cell == -1:
            return x

        if isinstance(x, torch.Tensor):
            x = [x]

        # blocks, 0 to n
        for i, m in enumerate(self.cells):
            if start_cell <= i:
                x = m(x)
            if end_cell == i:
                return x

        # head, otherwise
        return [self.get_last_head()(x[-1])]

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + [
            MetaArgument('cls_network_stem', Register.network_stems, help_name='network stem', allowed_num=1, optional_for_loading=True),
            MetaArgument('cls_network_heads', Register.network_heads, help_name='network heads', allow_duplicates=True, optional_for_loading=True),
            MetaArgument('cls_network_cells', Register.network_cells, help_name='network cells', allow_duplicates=True),
            MetaArgument('cls_network_cells_primitives', Register.primitive_sets, help_name='network cells primitives', allow_duplicates=True),
        ]

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('features_mul', default=1, type=int, help='global width multiplier for stem, cells'),
            Argument('features_first_cell', default=-1, type=int, help='fixed num output features of the first cell if >0'),
            Argument('cell_order', default="n, r, n", type=str, help='arrangement of cells'),
        ]

    @classmethod
    def search_network_from_args(cls, args: Namespace, index: int = None, weight_strategies: Union[dict, str] = None):
        """
        :param args: global argparse namespace
        :param index: index of this network
        :param weight_strategies: {strategy name: [cell indices]}, or name used for all, or None for defaults
        """
        all_args = cls._all_parsed_arguments(args)

        stem = cls._parsed_meta_argument(Register.network_stems, 'cls_network_stem', args, None).stem_from_args(args)
        heads = cls._parsed_meta_arguments(Register.network_heads, 'cls_network_heads', args, None)
        heads = nn.ModuleList([cls_head.head_from_args(args, index=i) for i, cls_head in enumerate(heads)])

        cells = cls._parsed_meta_arguments(Register.network_cells, 'cls_network_cells', args, None)
        primitives = cls._parsed_meta_arguments(Register.primitive_sets, 'cls_network_cells_primitives', args, None)
        assert len(cells) == len(primitives),\
            "Number of cells (%d) and primitives (%d) must match" % (len(cells), len(primitives))

        partial_cells = {}
        for i, (cell_cls, primitive_cls) in enumerate(zip(cells, primitives)):
            cell_name = cell_cls.get_name_in_args(args, index=i)
            assert issubclass(cell_cls, (SearchCellInterface, FixedSearchCellInterface))
            assert cell_name not in partial_cells, 'Can not have multiple cells with the same name!'
            assert issubclass(primitive_cls, PrimitiveSet)
            primitive = primitive_cls.from_args(args, index=i)
            partial_cells[cell_name] = cell_cls.partial_search_cell_instance(args=args, index=i, primitives=primitive)

        net = cls(stem=stem, heads=heads, cell_configs={}, cell_partials=partial_cells,
                  weight_strategies=weight_strategies, **all_args)
        return net

    # config, adding cells from config ---------------------------------------------------------------------------------

    def config(self, finalize=False, **__) -> dict:
        # update configs of (new) cells
        cell_configs = {}
        for cell, name in zip(self.cells, self.cell_order):
            if name in cell_configs.keys():
                continue
            cell_configs[name] = cell.config(finalize=finalize, **__)
        self.cell_configs.update(cell_configs)
        cfg = super().config(finalize=finalize, **__)
        return cfg

    def add_cells_from_config(self, config: dict):
        """ add all cell types in the given config """
        for name, cfg in config.get('kwargs').get('cell_configs').items():
            already_had = False
            if name in self._cell_partials.keys():
                already_had = True
                self._cell_partials.pop(name)
            if name in self.cell_configs.keys():
                already_had = True
            if already_had:
                LoggerManager().get_logger().info('%s cell type "%s" from given config' % ('Replaced' if already_had else 'Added', name))
            self.cell_configs[name] = cfg
