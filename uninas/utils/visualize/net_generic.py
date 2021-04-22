"""
generically visualize a network by tracing input+output Shapes
"""


from graphviz import Digraph
from collections import defaultdict
from uninas.modules.modules.abstract import AbstractModule
from uninas.utils.paths import replace_standard_paths
from uninas.utils.shape import Shape
from uninas.builder import Builder
from uninas.main import Main

run_config = '{path_conf_tasks}/s3.run_config'


class VizNode:
    shape_id = {}
    shape_id_as_input = defaultdict(list)
    shape_id_as_output = defaultdict(list)

    def __init__(self, o: AbstractModule, s_in, s_out, lst: list):
        self.id = str(id(o))
        self.o = o
        self.s_in = s_in
        self.s_out = s_out

        if len(lst) == 0:
            for s0, g in [(s_in, self.shape_id_as_input), (s_out, self.shape_id_as_output)]:
                if isinstance(s0, Shape):
                    g[s0.id].append(self)
                    self.shape_id[s0.id] = s0
                elif isinstance(s0, (tuple, list)):
                    for s1 in s0:
                        g[s1.id].append(self)
                        self.shape_id[s1.id] = s1
        else:
            pass

        self.nodes = [VizNode(*a) for a in lst]

    def print(self, indent=0):
        s_in_str = self.s_in.str() if self.s_in is not None else "None"
        s_out_str = self.s_out.str() if self.s_out is not None else "None"
        print('. '*indent, self.id, self.o.__class__.__name__, '\t', s_in_str, '\t', s_out_str)
        for n in self.nodes:
            n.print(indent=indent+1)

    def _plot(self, g: Digraph, add_subgraphs=True):
        if len(self.nodes) == 0:
            g.node(self.id, label=self.o.__class__.__name__)
        else:
            if add_subgraphs:
                with g.subgraph(name='cluster_%s' % self.id) as c:
                    c.attr(color='lightgrey', rank='same')
                    for n in self.nodes:
                        n._plot(c, add_subgraphs=add_subgraphs)
                    c.attr(label=self.o.__class__.__name__)
            else:
                for n in self.nodes:
                    n._plot(g, add_subgraphs=add_subgraphs)

    def plot(self, g: Digraph, add_subgraphs=True):
        g.attr(compound='true')
        self._plot(g, add_subgraphs=add_subgraphs)
        for s_id, inputs in self.shape_id_as_input.items():
            # print(s_id, len(self.shape_id_as_output.get(s_id, [])), len(inputs))
            for o in self.shape_id_as_output.get(s_id, []):
                for i in inputs:
                    g.edge(o.id, i.id, label=str(self.shape_id[s_id]))


class VizTree:
    def __init__(self, module: AbstractModule):
        self.module = module
        self.node = VizNode(*module.hierarchical_base_modules())

    def plot(self, path: str, add_subgraphs=True):
        g = Digraph('G', filename='plot.gv')
        self.node.plot(g, add_subgraphs=add_subgraphs)
        g.view(filename=path)

    def print(self):
        self.node.print(indent=0)


def visualize_config(config: dict, save_path: str):
    save_path = replace_standard_paths(save_path)
    cfg_path = Builder.save_config(config, replace_standard_paths('{path_tmp}/viz/'), 'viz')
    exp = Main.new_task(run_config, args_changes={
        '{cls_data}.fake': True,
        '{cls_data}.batch_size_train': 4,
        '{cls_task}.is_test_run': True,
        '{cls_task}.save_dir': '{path_tmp}/viz/task/',
        '{cls_task}.save_del_old': True,
        "{cls_network}.config_path": cfg_path,
    })
    net = exp.get_method().get_network()
    vt = VizTree(net)
    vt.print()
    vt.plot(save_path + 'net', add_subgraphs=True)
    print('Saved cell viz to %s' % save_path)


def visualize_file(config_path: str, save_dir: str):
    config_name = config_path.split('/')[-1].split('.')[0]
    save_path = '%s%s/' % (save_dir, config_name)
    config = Builder.load_config(config_path)
    visualize_config(config, save_path)


if __name__ == '__main__':
    visualize_file(Builder().find_net_config_path('MobileNetV2'), '{path_tmp}/viz/')
