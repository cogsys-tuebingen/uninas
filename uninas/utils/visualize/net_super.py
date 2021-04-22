"""
visualize a super network
"""


from graphviz import Digraph
from uninas.models.networks.uninas.search import SearchUninasNetwork
from uninas.modules.cells.single_layer import SingleLayerCell
from uninas.modules.heads.cnn import FeatureMixClassificationHead
from uninas.utils.generate.networks.super_configs import NetWrapper
from uninas.utils.paths import replace_standard_paths
from uninas.utils.misc import get_var_name
from uninas.builder import Builder

run_config = '{path_conf_tasks}/super1.run_config'


short_name = {
    'MobileNetV2Stem': 'Conv K3 + MB E3 K3',
    'MobileInvertedConvLayer': 'MB',
    'ShuffleNetV2Layer': 'SB',
    'ShuffleNetV2XceptionLayer': 'SX',
}
colors = {
    3: 'cyan2',
    5: 'palegreen',
    7: 'plum1',
    'misc': 'orange',
}


def width_str(expansion: str):
    return '%.2f' % ((4+int(expansion))/3)


def visualize_genotype(wrapper: NetWrapper, save_dir: str):
    Builder()
    config_name = get_var_name(wrapper)
    save_dir = replace_standard_paths('%s%s/' % (save_dir, config_name))
    wrapper_net, config, _ = wrapper.generate(save_dir, 'viz')
    assert isinstance(wrapper_net, SearchUninasNetwork)

    g = Digraph(format='pdf', engine='dot',
                edge_attr=dict(fontsize='20', fontname="times"),
                node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5',
                               penwidth='2', fontname="times"))
    cell_order = config.get('kwargs').get('cell_order')
    stem_name = wrapper_net.get_network().get_stem().__class__.__name__

    g.node('stem', label=short_name.get(stem_name, stem_name), width=width_str(expansion='3'),
           fillcolor=colors.get('misc'))
    node_names = ['stem']

    for i, cell in enumerate(wrapper_net.get_network().get_cells()):
        assert isinstance(cell, SingleLayerCell)
        name = cell_order[i]
        op_cfg = config.get('kwargs').get('cell_configs').get(name).get('submodules').get('op')
        cell_cls = op_cfg.get('name')
        op_kwargs = op_cfg.get('kwargs')

        e = op_kwargs.get('expansion')
        k = op_kwargs.get('k_size')
        s_in = cell.cached.get('shape_in')[0]
        s_out = cell.cached.get('shape_out')[0]

        label = '{name} E{e} K{k}'.format(**{
            'name': short_name.get(cell_cls, cell_cls),
            'e': e,
            'k': k,
        })
        g.node(name, label=label, width=width_str(expansion=e), fillcolor=colors.get(k))
        node_names.append(name)
        if len(node_names) > 1:
            g.edge(node_names[-2], node_names[-1], label='\t'+'*'.join([str(s) for s in s_in.shape]))
        print('{:<10}{:<30}{:<30}{:<30}{}'.format(cell.name, cell_cls, s_in.str(), s_out.str(), str(op_kwargs)))

    head = wrapper_net.get_network().get_heads()[-1]
    assert isinstance(head, FeatureMixClassificationHead)

    g.node('fmix', label='Conv K1', width=width_str(expansion='3'), fillcolor=colors.get('misc'))
    node_names.append('fmix')
    s_in = head.cached.get('shape_in')
    g.edge(node_names[-2], node_names[-1], label='\t'+'*'.join([str(s) for s in s_in.shape]))

    g.node('head', label='classification', width=width_str(expansion='3'), fillcolor=colors.get('misc'))
    node_names.append('head')
    s_in = head.cached.get('shape_inner')
    g.edge(node_names[-2], node_names[-1], label='\t'+'*'.join([str(s) for s in s_in.shape]))

    g.view(filename='%snet' % save_dir)
    print('Saved cell viz to %s' % save_dir)


if __name__ == '__main__':
    from uninas.utils.generate.networks.super_configs import FairNasC
    visualize_genotype(FairNasC, '{path_tmp}/viz/')
