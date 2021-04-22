"""
visualize the normal and reduction cell in a DARTS-like network
"""


from graphviz import Digraph
from uninas.modules.modules.abstract import AbstractModule
from uninas.modules.modules.misc import MultiModules
from uninas.modules.cells.darts import DartsCNNCell
from uninas.utils.paths import replace_standard_paths
from uninas.builder import Builder
from uninas.main import Main

run_config = '{path_conf_tasks}/s3.run_config'


def visualize_block(block, graph: Digraph, name: str, x: [str]):
    """ visualize the named edges of this block  """
    for i in range(len(block.ops)):
        in_idx = block.ops[i].module.idx
        stacked = ''
        op = block.ops[i].module.wrapped
        if isinstance(op, MultiModules):
            stacked = '%d*' % len(op.submodules)
            op = op.submodules[0]
        k = op.kwargs().get('k_size', None)
        type_ = op.kwargs().get('pool_type', None)
        dil = op.kwargs().get('dilation', 1)
        label = '{stacked}{class}({k}{type}{dil})'.format(**{
            'stacked': stacked,
            'class': op.__class__.__name__,
            'k': ('k=%d' % k) if k is not None else '',
            'type': (', %s' % type_) if type_ is not None else '',
            'dil': (', d=%d' % dil) if dil > 1 else ''
        })
        graph.edge(x[in_idx], name, label=label, fillcolor="gray")


def visualize_cell(cell: AbstractModule, save_path: str, name: str, show=True):
    assert isinstance(cell, DartsCNNCell)
    g = Digraph(format='pdf', engine='dot',
                edge_attr=dict(fontsize='20', fontname="times"),
                node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5',
                               width='0.5', penwidth='2', fontname="times"))
    g.body.extend(['rankdir=LR'])
    num_blocks = len(cell.blocks)
    states = ['%s_{k-2}', '%s_{k-1}'] + ['%s_' + str(i) for i in range(num_blocks)] + ['%s_{k}']
    states = [s % name for s in states]
    # input nodes
    for i in range(cell.num_inputs()):
        g.node(states[i], fillcolor='darkseagreen2', label=states[i].replace(name, 'c'))
    # blocks
    for i in range(cell.num_inputs(), cell.num_inputs() + num_blocks):
        g.node(states[i], fillcolor='lightblue', label=str(i))
    # output
    g.node(states[-1], fillcolor='palegoldenrod')
    for i in cell.concat.idxs:
        g.edge(states[i], states[-1], fillcolor="gray")
    for i, m in enumerate(cell.blocks):
        visualize_block(m, g, states[i+2], states)
    g.render('%s%s' % (save_path, name), view=show)


def visualize_config(config: dict, save_path: str):
    save_path = replace_standard_paths(save_path)
    cfg_path = Builder.save_config(config, replace_standard_paths('{path_tmp}/viz/'), 'viz')
    exp = Main.new_task(run_config, args_changes={
        '{cls_data}.fake': True,
        '{cls_data}.batch_size_train': 2,
        '{cls_task}.is_test_run': True,
        '{cls_task}.save_dir': '{path_tmp}/viz/task/',
        '{cls_task}.save_del_old': True,
        "{cls_task}.note": "viz",
        "{cls_network}.config_path": cfg_path,
    })
    net = exp.get_method().get_network()
    for s in ['n', 'r']:
        for cell in net.get_cells():
            if cell.name.startswith(s):
                visualize_cell(cell, save_path, s)
                break
    print('Saved cell viz to %s' % save_path)


def visualize_file(config_path: str, save_dir: str):
    config_name_ = Builder.net_config_name(config_path)
    save_path = save_dir+config_name_+'/'
    config = Builder.load_config(config_path)
    visualize_config(config, save_path)


if __name__ == '__main__':
    visualize_file(Builder().find_net_config_path('DARTS_V1'), '{path_tmp}/viz/')
