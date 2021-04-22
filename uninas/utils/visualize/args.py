"""
visualize the args tree
"""


from graphviz import Digraph
from uninas.main import Main
from uninas.utils.paths import replace_standard_paths
from uninas.utils.args import ArgsTreeNode, arg_list_from_json


colors = {
    'cls': 'cyan2',
    'meta': 'white',
}


def _visualize_args_tree(node: ArgsTreeNode, g: Digraph):
    g.node(node.name, label=node.name, fillcolor=colors.get('cls'))
    for meta_name, meta_children in node.children.items():
        if node.metas.get(meta_name):
            lim = node.metas[meta_name].limit_str()
            g.node(meta_name, label="%s (%s)" % (meta_name, lim), fillcolor=colors.get('meta'))
            g.edge(node.name, meta_name)
            for child in meta_children:
                _visualize_args_tree(child, g)
                g.edge(meta_name, child.name)


def visualize_args_tree(node: ArgsTreeNode):
    g = Digraph(format='pdf', engine='dot',
                edge_attr=dict(fontsize='20', fontname="times"),
                node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5',
                               penwidth='2', fontname="times"))
    _visualize_args_tree(node, g)
    return g


if __name__ == '__main__':
    from uninas.builder import Builder
    Builder()

    args_list = arg_list_from_json("{path_tmp}/s1/task.run_config")

    root = ArgsTreeNode(Main)
    root.build_from_args(args_list)
    print("-"*200)
    visualize_args_tree(root).view(filename="args_tree", directory=replace_standard_paths("{path_tmp}"),
                                   cleanup=True, quiet_view=True)
