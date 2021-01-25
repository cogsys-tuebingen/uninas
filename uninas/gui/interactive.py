"""
backend for the gui, interactively add/remove argparse nodes
"""


import typing
from uninas.utils.args import MetaArgument, ArgsTreeNode, ArgumentParser,\
    arg_list_from_json, save_as_json, replace_wildcards_in_args_list
from uninas.utils.meta import Singleton
from uninas.main import Main
from uninas.builder import Builder


class EventHook:
    def on_add_child(self, node, child, meta: MetaArgument):
        print(' > adding child', node.name, child.name)
        child.add_hook(self)

    def on_update_indices(self, node, meta: MetaArgument):
        print(' > updating indices', node.name, meta.argument.name)

    def on_delete(self, node):
        print(' > deleting', node.name)

    def on_delete_child(self, node, child, meta_name: str, index: int):
        print(' > deleting child', node.name, child.name, meta_name, index)


class GuiArgsTreeNode(ArgsTreeNode):
    """
    extends the ArgsTreeNode by some functions that are necessary for a GUI,
    such as storing and updating argument values or changing the meta structure
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooks = []
        self._arg_values = {arg.name: arg.default for arg in self.args_cls.args_to_add(index=None)}

    def add_hook(self, hook: EventHook):
        self.hooks.append(hook)

    def _add_child(self, meta: MetaArgument, child):
        super()._add_child(meta, child)
        for hook in self.hooks:
            hook.on_add_child(self, child, meta=meta)

    def _update_indices(self, meta: MetaArgument):
        super()._update_indices(meta)
        for hook in self.hooks:
            hook.on_update_indices(self, meta)

    def _can_parse(self, args_list: [str], parser: ArgumentParser):
        """ update current values """
        args_list, _ = replace_wildcards_in_args_list(args_list, self.get_wildcards())
        for arg_str in args_list:
            s0, s1 = arg_str.split('=')
            name, value = s0.replace('--', ''), s1
            if name.startswith(self.name):
                key = name.split('.')[1]
                self._arg_values[key] = value

    def set_arg_value(self, name: str, value):
        self._arg_values[name] = value

    def get_arg_value(self, name: str):
        return self._arg_values.get(name)

    def get_arg_values(self) -> dict:
        return self._arg_values

    def yield_args(self) -> typing.Iterable[tuple]:
        """ get argument and their currently stored value """
        for arg in self.args_cls.args_to_add(index=None):
            yield arg, self._arg_values.get(arg.name)

    def yield_meta_args(self) -> typing.Iterable[tuple]:
        """ get (meta arg, meta arg values) """
        for meta in self.args_cls.meta_args_to_add():
            yield meta, [c.name for c in self.children[meta.argument.name]]

    def get_meta_by_name(self, meta_name: str) -> MetaArgument:
        """ get meta with the given name """
        for meta in self.args_cls.meta_args_to_add():
            if meta.argument.name == meta_name:
                return meta

    def yield_nodes(self) -> typing.Iterable:
        """ recursively yield through all child nodes including this one """
        yield self
        for children in self.children.values():
            for child in children:
                for cx in child.yield_nodes():
                    yield cx

    def on_delete(self):
        for hook in self.hooks:
            hook.on_delete(self)

    def remove_child(self, meta_name: str, index=0):
        """ remove from a meta argument """
        if len(self.children[meta_name]) > index:
            child = self.children[meta_name].pop(index)
            child.on_delete()
            for hook in self.hooks:
                hook.on_delete_child(self, child, meta_name, index)
            self._update_indices(self.metas[meta_name])


class Interactive(metaclass=Singleton):
    def __init__(self, hooks=()):
        Builder()
        self.root = GuiArgsTreeNode(Main)
        for hook in hooks:
            self.root.add_hook(hook)

    def add_root_hook(self, hook: EventHook):
        self.root.add_hook(hook)

    def from_json(self, paths: str):
        """ read a config file, update node structure and defaults accordingly """
        parser = ArgumentParser("UniNAS GUI")
        arg_list = arg_list_from_json(paths)
        self.root.build_from_args(arg_list, parser=parser, raise_problems=False)
        self.root.parse(arg_list, parser, raise_unparsed=False)

    def to_json(self, path: str):
        """ export the current args to a config file """
        _, wildcards, _, _ = self.root.parse([], None, raise_unparsed=False)
        save_as_json(self.get_args_dict(), path, wildcards=wildcards)

    def get_node(self, name: str) -> GuiArgsTreeNode:
        """ get a node by name """
        for node in self.root.yield_nodes():
            if node.name == name:
                return node

    def add_meta_value(self, node_name: str, meta_name: str, meta_value: str):
        node = self.get_node(node_name)
        meta_cls = node.get_meta_by_name(meta_name)
        node.add_child_meta(meta_cls, meta_value)

    def remove_meta_index(self, node_name: str, meta_name: str, meta_index: int = None):
        node = self.get_node(node_name)
        node.remove_child(meta_name, index=meta_index)

    def set_arg_value(self, node_name: str, arg_name: str, arg_value: str):
        node = self.get_node(node_name)
        node.set_arg_value(arg_name, arg_value)

    def yield_meta_args(self) -> typing.Iterable[tuple]:
        """ get (node, meta arg, meta arg values) of all args """
        for node in self.root.yield_nodes():
            for meta, names in node.yield_meta_args():
                yield node, meta, names

    def yield_args(self) -> typing.Iterable[tuple]:
        """ get (node, arg, cur arg value) of all args """
        for node in self.root.yield_nodes():
            for arg, cur in node.yield_args():
                yield node, arg, cur

    def get_args_dict(self) -> [str]:
        """ get a dict of all current (meta) args """
        dct = {}
        for node, meta, names in self.yield_meta_args():
            dct["%s" % meta.argument.name] = "%s" % ', '.join([n.split('#')[0] for n in names])
        for node, arg, cur in self.yield_args():
            dct["%s.%s" % (node.name, arg.name)] = cur
        return dct

    def print_meta_args(self):
        print('-'*100)
        for node, meta, names in self.yield_meta_args():
            print(node.name, meta.argument.name, names)
        print('-'*100)

    def print_args(self):
        print('-'*100)
        for node, arg, cur in self.yield_args():
            print(node.name, arg.name, cur, arg.default)
        print('-'*100)


def exp():
    interactive = Interactive(hooks=[EventHook()])
    # interactive.from_json("{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}/fairnas.run_config")
    interactive.from_json("{path_conf_tasks}/s1_fairnas.run_config")
    interactive.from_json("{path_conf_net_search}/fairnas.run_config")
    interactive.print_meta_args()

    # making some changes
    interactive.set_arg_value('StackedCellsNetworkBody', 'cell_order', 'the cell order is now useless!')
    interactive.remove_meta_index('StrictlyFairRandomMethod', 'cls_data', meta_index=0)
    interactive.add_meta_value('StrictlyFairRandomMethod', 'cls_data', 'Imagenet1000Data')
    interactive.add_meta_value('Imagenet1000Data', 'cls_augmentations', 'CutoutAug')
    interactive.add_meta_value('Imagenet1000Data', 'cls_augmentations', 'AACifar10Aug')

    interactive.print_args()
    interactive.to_json("{path_tmp}/test.run_config")


if __name__ == '__main__':
    exp()
