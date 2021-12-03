"""
Convenience wrapper for a NAS network,
- bundle functions for arc_params / net_params
- automatically call WeightStrategy(s) each forward pass, to update / sample architecture weights
- saving + restoring this also keeps trained architecture weights
- track used parameters/buffers depending on architecture indices
"""


from typing import Union
import torch
import torch.nn as nn
from uninas.modules.networks.abstract import AbstractNetworkBody
from uninas.methods.abstract_strategy import AbstractWeightStrategy
from uninas.methods.strategy_manager import StrategyManager
from uninas.models.networks.uninas.abstract import AbstractUninasNetwork
from uninas.utils.args import MetaArgument, Namespace
from uninas.utils.shape import Shape, ShapeList
from uninas.register import Register


def get_to_print(module: nn.Module) -> list:
    return list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))


class TrackerModuleEntry:
    def __init__(self, name: str, shape, req_grad: bool):
        self.name = name
        self.shape = shape
        self.req_grad = req_grad

    def numel(self) -> int:
        return self.shape.numel()

    def __str__(self) -> str:
        return '{:<5} {:<20} {:20}'.format(str(self.req_grad), str(list(self.shape)), self.name)


class TrackerCellEntry:
    def __init__(self):
        self.entries = []

    def get_entries(self) -> [TrackerModuleEntry]:
        return self.entries

    def append_module(self, name: str, module: nn.Module):
        for i, (n, p) in enumerate(get_to_print(module)):
            req_grad = p.requires_grad if isinstance(p, nn.Parameter) else False
            entry = TrackerModuleEntry('%s.%s' % (name, n), p.shape, req_grad)
            self.entries.append(entry)

    def finalize(self):
        """ remove duplicates """
        to_remove = []
        names = [entry.name for entry in self.get_entries()]
        for ck, entry in enumerate(self.get_entries()):
            if entry.name in names[:ck]:
                to_remove.append(ck)
        for tr in reversed(to_remove):
            self.entries.pop(tr)


class Tracker:
    """
    track parameter usage for one forward pass
    """

    def __init__(self):
        self.cells = []

    def get_cells(self) -> [TrackerCellEntry]:
        return self.cells

    def on_hook(self, cell_idx: int, name: str, module: nn.Module):
        while len(self.cells) < cell_idx+1:
            self.cells.append(TrackerCellEntry())
        self.cells[cell_idx].append_module(name, module)

    def finalize(self):
        """ remove duplicates per choice, if a choice had to be selected multiple times """
        for tracker_cell_entry in self.get_cells():
            tracker_cell_entry.finalize()

    def print(self):
        for ci, tracker_cell_entry in enumerate(self.get_cells()):
            print('cell %d, used params' % ci)
            for entry in tracker_cell_entry.get_pareto_best():
                print('{:4} {}'.format('', str(entry)))
            print('-'*30)


class Hook:
    """
    when a module is executed, inform the tracker
    """

    def __init__(self, tracker: Tracker, name: str):
        self.tracker = tracker
        self.cell_idx = int(name.split('cells.')[1].split('.')[0])
        self.name = name

    def __call__(self, module: nn.Module, module_input, module_output):
        self.tracker.on_hook(self.cell_idx, self.name, module)


@Register.network(search=True)
class SearchUninasNetwork(AbstractUninasNetwork):

    def __init__(self, model_name: str, net: AbstractNetworkBody, do_forward_strategy=True, *args, **kwargs):
        super().__init__(model_name=model_name, net=net, *args, **kwargs)
        self.do_forward_strategy = do_forward_strategy  # unnecessary line to remove "error" highlighting
        self._add_to_kwargs(do_forward_strategy=self.do_forward_strategy)
        self.strategy_manager = StrategyManager()
        self.strategies = None

    @classmethod
    def from_args(cls, args: Namespace, index=None, weight_strategies: Union[dict, str] = None)\
            -> 'SearchUninasNetwork':
        """
        :param args: global argparse namespace
        :param index: argument index
        :param weight_strategies: {strategy name: [cell indices]}, or name used for all, or None for defaults
        """
        all_parsed = cls._all_parsed_arguments(args)
        cls_net = cls._parsed_meta_argument(Register.network_bodies, 'cls_network_body', args, index=index)
        net = cls_net.search_network_from_args(args, index=index, weight_strategies=weight_strategies)
        return cls(cls.__name__, net, **all_parsed)

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + [
            MetaArgument('cls_network_body', Register.network_bodies, help_name='network', allowed_num=1),
        ]

    def _build2(self, s_in: Shape, s_out: Shape) -> ShapeList:
        """ build the network """
        s = self.net.build(s_in, s_out)
        self.strategies = self.strategy_manager.get_strategies_list()
        self.strategy_manager.build()
        return s

    def get_strategy_manager(self) -> StrategyManager:
        return self.strategy_manager

    def set_forward_strategy(self, forward_strategy: bool):
        self.do_forward_strategy = forward_strategy

    def get_forward_strategy(self) -> bool:
        return self.do_forward_strategy

    def forward(self, x: torch.Tensor, ws_kwargs: dict = None, **net_kwargs) -> [torch.Tensor]:
        """
        forward first the weight strategy, then the network
        """
        if self.do_forward_strategy:
            self.forward_strategy(**({} if ws_kwargs is None else ws_kwargs))
        return super().forward(x, **net_kwargs)

    def forward_net(self, x: torch.Tensor, **net_kwargs) -> [torch.Tensor]:
        """
        forward only the network
        """
        return self.net(x, **net_kwargs)

    def forward_strategy(self, **ws_kwargs):
        """
        forward only the weight strategy
        """
        self.strategy_manager.forward(**ws_kwargs)

    def str(self, depth=0, **_) -> str:
        r = '{d}{name}(\n{ws},{net}\n{d}])'.format(**{
            'd': '{d}',
            'd1': '{d1}',
            'name': self.__class__.__name__,
            'ws': '{d1}Strategies: [%s]' % ', '.join([ws.str() for ws in self.strategies]),
            'net': self.net.str(depth=depth+1, max_depth=self.log_detail, **_),
        })
        r = r.replace('{d}', '. '*depth).replace('{d1}', '. '*(depth+1))
        return r

    def config(self, finalize=True, **_) -> dict:
        if finalize:
            return self.net.config(finalize=finalize, **_)
        return super().config(finalize=finalize, **_)

    def named_net_arc_parameters(self) -> (list, list):
        # all named parameters
        net_params = list(self.net.named_parameters())
        arc_params = list(self.strategy_manager.named_parameters())
        duplicate_idx = []
        # remove arc parameters from the network
        for an, ap in arc_params:
            for idx, (n, p) in enumerate(net_params):
                if ap is p:
                    duplicate_idx.append(idx)
        for idx in sorted(duplicate_idx, reverse=True):
            net_params.pop(idx)
        return net_params, arc_params

    def track_used_params(self, x: torch.Tensor) -> Tracker:
        """
        track which weights are used for the current architecture,
        and in which cell
        """
        tracker = Tracker()
        is_train = self.training
        self.eval()
        handles = []
        ws_modules = []
        x = x.to(self.get_device())

        # find all modules that have a weight strategy, add hooks
        for name, module in self.named_modules():
            if hasattr(module, 'ws') and isinstance(module.ws, (AbstractWeightStrategy, StrategyManager)):
                ws_modules.append(module)
                for name2, m2 in module.named_modules():
                    if len(get_to_print(m2)) >= 1:
                        handles.append(m2.register_forward_hook(Hook(tracker, 'net.%s.%s' % (name, name2))))

        # forward pass with the current arc, all used weights are tracked
        self.forward_net(x)

        tracker.finalize()
        for h in handles:
            h.remove()
        self.train(is_train)
        return tracker

    @classmethod
    def get_space_tuple(cls, unique=True, flat=False) -> tuple:
        """ tuple of final topology """
        return tuple(StrategyManager().get_all_finalized_indices(unique=unique, flat=flat))
