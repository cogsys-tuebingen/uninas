"""
Convenience wrapper for a NAS network,
- bundle functions for arc_params / net_params
- automatically call WeightHelper(s) each forward pass, to update / sample architecture weights
- saving + restoring this also keeps trained architecture weights
- track used parameters/buffers depending on architecture indices
"""


from typing import Union
import torch
import torch.nn as nn
from uninas.model.networks.abstract import AbstractNetworkBody
from uninas.methods.strategies.abstract import AbstractWeightStrategy
from uninas.methods.strategies.manager import StrategyManager
from uninas.networks.self.abstract import AbstractUninasNetwork
from uninas.utils.args import MetaArgument, Namespace
from uninas.utils.shape import Shape
from uninas.register import Register


def get_to_print(module: nn.Module) -> list:
    return list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))


class Tracker:
    def __init__(self):
        self.arc = []    # net architecture, ints
        self.cells = []  # [cell idx][arc idx][(name, shape, req grad)]

    def set_arc(self, arc: [int]):
        self.arc = arc

    def on_hook(self, cell_idx: int, name: str, module: nn.Module):
        if len(self.arc) <= cell_idx:
            return
        while len(self.cells) < cell_idx+1:
            self.cells.append([])
        while len(self.cells[cell_idx]) < self.arc[cell_idx]+1:
            self.cells[cell_idx].append([])
        for i, (n, p) in enumerate(get_to_print(module)):
            req_grad = p.requires_grad if isinstance(p, nn.Parameter) else False
            self.cells[cell_idx][self.arc[cell_idx]].append(('%s.%s' % (name, n), p.shape, req_grad))

    def finalize(self):
        """ remove duplicates per choice, if a choice had to be selected multiple times """
        for ci, cell in enumerate(self.cells):
            for cj, choice in enumerate(cell):
                to_remove = []
                if len(choice) > 0:
                    names, shapes, grads = zip(*choice)
                    for ni, n in enumerate(names):
                        if n in names[:ni]:
                            to_remove.append(ni)
                    for tr in reversed(to_remove):
                        choice.pop(tr)

    def print(self):
        for ci, cell in enumerate(self.cells):
            print('cell %d' % ci)
            for cj, choice in enumerate(cell):
                print(' '*2, 'choice %d' % cj)
                for n, p, r in choice:
                    print('{:4} {:<5} {:<20} {:20}'.format('', str(r), str(list(p)), n))
            print('-'*30)


class Hook:
    def __init__(self, tracker: Tracker, name: str):
        self.tracker = tracker
        self.cell_idx = int(name.split('.')[3])
        self.name = name

    def __call__(self, module: nn.Module, _, __):
        self.tracker.on_hook(self.cell_idx, self.name, module)


@Register.network(search=True)
class SearchUninasNetwork(AbstractUninasNetwork):

    def __init__(self, name: str, net: AbstractNetworkBody, do_forward_strategy=True, **_):
        super().__init__(name, net, **_)
        self.do_forward_strategy = do_forward_strategy  # unnecessary line to remove "error" highlighting
        self._add_to_kwargs(do_forward_strategy=self.do_forward_strategy)
        self.strategy_manager = StrategyManager()
        self.strategies = None
        self.tracker = Tracker()

    @classmethod
    def from_args(cls, args: Namespace, index=None, weight_strategies: Union[dict, str] = "default"):
        """
        :param args: global argparse namespace
        :param index: argument index
        :param weight_strategies: {strategy name: [cell indices]}, or name used for all
        """
        all_parsed = cls._all_parsed_arguments(args)
        cls_net = cls._parsed_meta_argument('cls_network_body', args, index=index)
        net = cls_net.search_network_from_args(args, weight_strategies=weight_strategies)
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

    def _build2(self, s_in: Shape, s_out: Shape) -> Shape:
        s = self.net.build(s_in, s_out)
        self.strategies = self.strategy_manager.get_strategies_list()
        self.strategy_manager.build()
        self._track_paths(s_in.random_tensor(batch_size=2))
        return s

    def get_strategy_manager(self) -> StrategyManager:
        return self.strategy_manager

    def set_forward_strategy(self, forward_strategy: bool):
        self.do_forward_strategy = forward_strategy

    def forward(self, x: torch.Tensor, ws_kwargs: dict = None, **net_kwargs) -> [torch.Tensor]:
        if self.do_forward_strategy:
            self.forward_strategy(**({} if ws_kwargs is None else ws_kwargs))
        return self.net(x, **net_kwargs)

    def forward_net(self, x: torch.Tensor, **net_kwargs) -> [torch.Tensor]:
        return self.net(x, **net_kwargs)

    def forward_strategy(self, **ws_kwargs):
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
        net_params, arc_params, duplicate_idx = list(self.net.named_parameters()), [], []
        for ws in self.strategies:
            arc_params += list(ws.named_parameters())
        # remove arc parameters from the network
        for an, ap in arc_params:
            for idx, (n, p) in enumerate(net_params):
                if ap is p:
                    duplicate_idx.append(idx)
        for idx in sorted(duplicate_idx, reverse=True):
            net_params.pop(idx)
        return net_params, arc_params

    def _track_paths(self, x: torch.Tensor):
        """
        called once after building the network
        track which weights are used depending on which architecture
        note that this does NOT work with strategies that activate multiple paths at once
        """
        handles = []
        ws_modules = []

        # sample input to correct device
        for p in self.parameters():
            x = x.to(p.device)
            break

        # find all modules that have a weight strategy, add hooks
        for name, module in self.named_modules():
            if hasattr(module, 'ws') and isinstance(module.ws, (AbstractWeightStrategy, StrategyManager)):
                ws_modules.append(module)
                for name2, m2 in module.named_modules():
                    if len(get_to_print(m2)) >= 1:
                        handles.append(m2.register_forward_hook(Hook(self.tracker, 'net.%s.%s' % (name, name2))))

        # make several forward passes, track which modules were used depending on which arc
        arc_max = StrategyManager().ordered_num_choices()
        for i in range(max(arc_max)):
            arc = [min(i, m - 1) for m in arc_max]
            self.tracker.set_arc(arc)
            self.forward_strategy(fixed_arc=arc)
            self.forward_net(x)

        self.tracker.finalize()
        for h in handles:
            h.remove()
        # self.tracker.print()

    def save_to_state_dict(self) -> dict:
        """ store additional info in the state dict """
        return dict(cells=self.tracker.cells)

    def get_space_tuple(self, unique=True, flatten=False) -> tuple:
        """ tuple of final topology """
        st = StrategyManager().get_all_finalized_indices(unique=unique)
        if flatten:
            st2 = []
            for s in st:
                if isinstance(s, int):
                    st2.append(s)
                elif isinstance(s, (tuple, list)):
                    assert len(s) == 1, "can not flatten %s into an int" % str(s)
                    st2.append(s[0])
            st = st2
        return tuple(st)
