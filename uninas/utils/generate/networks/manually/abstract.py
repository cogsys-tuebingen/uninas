from functools import partial
from typing import Type
import torch.nn as nn
from uninas.modules.modules.abstract import AbstractModule
from uninas.modules.stems.abstract import AbstractStem
from uninas.modules.heads.abstract import AbstractHead
from uninas.modules.networks.abstract import AbstractNetworkBody
from uninas.modules.networks.stackedcells import StackedCellsNetworkBody
from uninas.modules.cells.single_layer import SingleLayerCell
from uninas.utils.shape import Shape


def get_stem_instance(cls: Type[AbstractStem], **diff_kwargs) -> AbstractStem:
    cls_kwargs = cls.parsed_argument_defaults()
    cls_kwargs.update(diff_kwargs)
    instance = cls.stem_from_kwargs(**cls_kwargs)
    assert isinstance(instance, AbstractStem)
    return instance


def get_head_instance(cls: Type[AbstractHead], **diff_kwargs) -> AbstractHead:
    cls_kwargs = cls.parsed_argument_defaults()
    cls_kwargs.update(diff_kwargs)
    instance = cls.head_from_kwargs(**cls_kwargs)
    assert isinstance(instance, AbstractHead)
    return instance


def get_passthrough_partial(layer: AbstractModule, **diff_kwargs):
    cell = SingleLayerCell(op=layer, **diff_kwargs)
    return lambda cell_idx: cell


def get_passthrough_partials(layers: [(int, AbstractModule, dict, dict)]) -> (dict, list):
    def for_partial(op: AbstractModule, *_, **__):
        return op

    cells, cell_order, last_features = {}, [], None
    for i, (num_features, op_cls, op_kwargs1, op_kwargs2) in enumerate(layers):
        # get the operand / layer
        op_kwargs = op_kwargs1.copy()
        op_kwargs.update(op_kwargs2)
        op = op_cls(**op_kwargs)
        # wrap it
        name = '%d-s%d' % (i, op.stride)
        if num_features == last_features:
            cell_kwargs = dict(name=name, features_mult=1, features_fixed=-1)
        else:
            cell_kwargs = dict(name=name, features_mult=-1, features_fixed=num_features)
            last_features = num_features
        cells[name] = partial(for_partial, op=SingleLayerCell(op=op, **cell_kwargs))
        cell_order.append(name)
    return cells, cell_order


def get_network(net_cls: Type[AbstractNetworkBody], stem: AbstractModule, head: AbstractModule,
                cell_partials: dict, cell_order: [str],
                s_in=Shape([3, 224, 224]), s_out=Shape([1000])) -> nn.Module:
    net_kwargs = net_cls.parsed_argument_defaults()
    net_kwargs.update(dict(cell_configs={}, cell_partials=cell_partials, cell_order=cell_order))
    network = StackedCellsNetworkBody(stem=stem, heads=nn.ModuleList([head]), **net_kwargs)
    network.build(s_in=s_in, s_out=s_out)
    return network
