import torch
import torch.nn as nn
from typing import Union
from collections import OrderedDict
from uninas.model.networks.abstract import AbstractNetworkBody
from uninas.model.modules.abstract import AbstractModule
from uninas.model.cells.abstract import AbstractCell, SearchCellInterface, FixedSearchCellInterface
from uninas.utils.misc import split
from uninas.utils.shape import Shape
from uninas.utils.args import MetaArgument, Argument, Namespace
from uninas.utils.loggers.python import get_logger
from uninas.register import Register

logger = get_logger()


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
        :param weight_strategies: {strategy name: [cell indices]}, or name used for all, or None
        :param kwargs_to_save:
        """
        super().__init__(**kwargs_to_save)
        self._add_to_submodules(stem=stem)
        self._add_to_submodule_lists(heads=heads)
        self._add_to_kwargs_np(cell_configs=cell_configs, cell_partials={})
        if isinstance(cell_order, str):
            cell_order = split(cell_order)
        self._add_to_kwargs(cell_order=cell_order, weight_strategies=weight_strategies)
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

    def set_dropout_rate(self, p=None, stem_p=None, cells_p=None):
        """ set the dropout rate of every dropout layer to p, no change for p=None, only change the head by default """
        self.stem.set_dropout_rate(stem_p)
        for m in self.cells:
            m.set_dropout_rate(cells_p)
        for m in self.heads:
            m.set_dropout_rate(p)

    def get_head_weightings(self) -> [float]:
        """ get the weights of all heads, in order (the last head at -1 has to be last) """
        return [v.weight for v in self._head_positions.values()]

    def _build(self, s_in: Shape, s_out: Shape) -> Shape:
        log_str = '  {:<8}{:15}{:<32}{:<45}-> {:<45}'
        logger.info('Building %s:' % self.__class__.__name__)
        logger.info(log_str.format('index', 'name', 'class', 'input shapes', 'output shapes'))
        s_out_data = s_out.copy()
        out_shapes = self.stem.build(s_in)
        self._build_log_layer(log_str, '', '-', self.stem)

        # cells and (aux) heads
        updated_cell_order = []
        for i, cell_name in enumerate(self.cell_order):
            cell = self._get_cell(name=cell_name, cell_index=i)
            assert self.stem.num_outputs() == cell.num_inputs() == cell.num_outputs(), 'Cell does not fit the network!'
            updated_cell_order.append(cell.name)
            s_ins = out_shapes[-cell.num_inputs():]
            s_out = cell.build(s_ins.copy(),
                               features_mul=self.features_mul,
                               features_fixed=self.features_first_cell if i == 0 else -1)
            out_shapes.extend(s_out)
            self._build_log_layer(log_str, i, cell_name, cell)
            self.cells.append(cell)

            # optional (aux) head after every cell
            head = self._head_positions.get(i, None)
            if head is not None:
                if head.weight > 0:
                    head.build(s_out[-1], s_out_data)
                    self._build_log_layer(log_str, '', '-', head)
                else:
                    logger.info('not adding head after cell %d, weight <= 0' % i)
                    del self._head_positions[i]
            else:
                assert i != len(self.cell_order) - 1, "Must have a head after the final cell"

        # remove heads that are impossible to add
        for i in self._head_positions.keys():
            if i >= len(self.cells):
                logger.warning('Can not add a head after cell %d which does not exist, deleting the head!' % i)
                head = self._head_positions.get(i)
                for j, head2 in enumerate(self.heads):
                    if head is head2:
                        self.heads.__delitem__(j)
                        break
        self.set(cell_order=updated_cell_order)
        return s_out

    @staticmethod
    def _build_log_layer(log_str: str, idx: str, name: str, obj: AbstractModule):
        s_in_str = obj.get_cached('shape_in').str()
        s_inner = obj.get_cached('shape_inner')
        if s_inner is not None:
            s_in_str = '{:20} -> {}'.format(s_in_str, s_inner.str())
        s_out_str = obj.get_cached('shape_out').str()
        logger.info(log_str.format(idx, name, obj.__class__.__name__, s_in_str, s_out_str))

    def _get_cell(self, name: str, cell_index: int) -> AbstractCell:
        """ get a cell, either from known config or an already partially built one """
        if name in self._cell_partials:
            if self.weight_strategies is None:
                return self._cell_partials[name](cell_index=cell_index)
            elif isinstance(self.weight_strategies, str):
                return self._cell_partials[name](cell_index=cell_index, strategy_name=self.weight_strategies)
            elif isinstance(self.weight_strategies, dict):
                for k, v in self.weight_strategies.items():
                    if cell_index in v:
                        return self._cell_partials[name](cell_index=cell_index, strategy_name=k)
                raise KeyError("missing name of weight strategy at cell index %d" % cell_index)
            raise NotImplementedError("unknown format for weight strategies: %s" % type(self.weight_strategies))
        if name in self.cell_configs:
            return Register.builder.from_config(self.cell_configs[name])
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
    def search_network_from_args(cls, args: Namespace, weight_strategies: Union[dict, str] = 'default'):
        """
        :param args: global argparse namespace
        :param weight_strategies: {strategy name: [cell indices]}, or name used for all
        """
        all_args = cls._all_parsed_arguments(args)

        stem = cls._parsed_meta_argument('cls_network_stem', args, None).stem_from_args(args)
        heads = nn.ModuleList([cls_head.head_from_args(args, index=i)
                               for i, cls_head in enumerate(cls._parsed_meta_arguments('cls_network_heads', args, None))])

        partial_cells = {}
        for i, cell_cls in enumerate(cls._parsed_meta_arguments('cls_network_cells', args, None)):
            cell_name = cell_cls.get_name_in_args(args, index=i)
            assert issubclass(cell_cls, (SearchCellInterface, FixedSearchCellInterface))
            assert cell_name not in partial_cells, 'Can not have multiple cells with the same name!'
            partial_cells[cell_name] = cell_cls.partial_search_cell_instance(args=args, index=i)

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
        # remove strategy names when finalizing
        cfg = super().config(finalize=finalize, **__)
        if finalize:
            s = cfg.get("kwargs").pop("weight_strategies")
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
                logger.info('%s cell type "%s" from given config' % ('Replaced' if already_had else 'Added', name))
            self.cell_configs[name] = cfg


@Register.network_body()
class StudentStackedCellsNetworkBody(StackedCellsNetworkBody):
    """
    A model made from stacking cells
    Purely sequential, one stem with n outputs, every cell has n inputs and n outputs
    For knowledge distillation only, limited to one head
    """

    def _update_heads(self):
        super()._update_heads()
        assert len(self.heads) == 1, 'Can not have multiple heads!'

    @classmethod
    def is_student_network(cls) -> bool:
        """ for knowledge distillation """
        return True

    def forward(self, x: Union[torch.Tensor, list], start_block=-1, end_block=None) -> [torch.Tensor]:
        """
        can execute specific part of the network, returns result after end_block
        """

        # stem, -1
        if start_block <= -1:
            x = self.stem(x)
        if end_block == -1:
            return x

        if isinstance(x, torch.Tensor):
            x = [x]

        # blocks, 0 to n
        for i, m in enumerate(self.cells):
            if start_block <= i:
                x = m(x)
            if end_block == i:
                return x

        # head, otherwise
        return [self.get_last_head()(x[-1])]
