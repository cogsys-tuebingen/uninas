import os
import tkinter as tk
import tkinter.messagebox as tkm
import tkinter.ttk as ttk
from tkinter import filedialog
from uninas.main import Main
from uninas.register import Register
from uninas.utils.args import MetaArgument, Argument
from uninas.utils.paths import standard_paths, replace_standard_paths, get_class_path, name_task_config
from uninas.utils.loggers.python import LoggerManager
from uninas.utils.visualize.args import visualize_args_tree
from uninas.gui.interactive import Interactive, GuiArgsTreeNode, EventHook
from uninas.gui.tk_gui.utils.toggled import ToggledFrame
from uninas.gui.tk_gui.utils.tooltip import CreateToolTip


sizes = {
    'borderwidth': 4,
    'borderwidth_entry': 2,
    'wrap_tooltip': 400,
    'label_entry_width': 180,
    'label_entry_height': 20,
}
colors = {
    'background_default': 'lightgrey',
    'background_highlight': 'blue',
    'meta_count_correct': 'DarkSeaGreen1',
    'meta_count_unclear': 'gold',
    'meta_count_wrong': 'red',
}
misc = {
    'scroll_delta': 2,
}


def maybe_add_cls_tooltip(name: str, label: tk.Label = None, tooltip: CreateToolTip = None):
    cls_name = name.split('#')[0]
    try:
        cls = Register.get(cls_name)
        if cls is not None:
            text = ''
            if cls.__doc__ is not None and len(cls.__doc__) > 0:
                for i, line in enumerate(cls.__doc__.split('\n')):
                    if i == len(line) == 0:
                        continue
                    text += line.replace('    ', '', 1) + '\n'
                text += '\n\n'
            text += '(implemented in: %s)' % get_class_path(cls)
            if tooltip is None:
                CreateToolTip(label, text=text, wraplength=sizes.get('wrap_tooltip'))
            else:
                tooltip.text = text
    except:
        pass


class UpdatableFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, bg=colors.get('background_default'))
        self._config_enabled = False

    @classmethod
    def yield_updatable_children(cls, frame):
        if isinstance(frame, UpdatableFrame):
            yield frame
        for child in frame.winfo_children():
            for u in cls.yield_updatable_children(child):
                yield u

    def enable_config_changes(self, b=True):
        for child in self.yield_updatable_children(self):
            child._config_enabled = b

    def update_content(self):
        for child in self.yield_updatable_children(self):
            child._update_content()

    def _update_content(self):
        pass

    def highlight_content(self, s: str) -> int:
        s = s if (s is not None and len(s) > 0) else '__no__highlight__'
        s = s.lower()
        return sum([child._highlight_content(s) for child in self.yield_updatable_children(self)])

    def _highlight_content(self, s: str) -> int:
        return 0


class ArgsEntryFrame(UpdatableFrame):
    """
    one key: value pair
    """

    def __init__(self, master, interactive: Interactive, node: GuiArgsTreeNode, arg: Argument, *args, **kwargs):
        super().__init__(master, *args, **kwargs, borderwidth=sizes.get('borderwidth_entry'))
        self.interactive = interactive
        self.node = node
        self.arg = arg

        self.label_frame = tk.Frame(self, height=sizes.get('label_entry_height'),
                                    width=sizes.get('label_entry_width'), bg=colors.get('background_default'))
        self.label_frame.pack_propagate(0)
        self.label_frame.pack(side=tk.LEFT)
        self.label = tk.Label(self.label_frame, text=arg.name, bg=colors.get('background_default'))
        self.label.pack(side=tk.LEFT)

        # label and variable
        value = arg.apply_rules(self.node.get_arg_value(self.arg.name))
        if arg.is_bool:
            self.var = tk.IntVar(value=1 if value else 0)
            mini_frame = tk.Frame(self, bg=colors.get('background_default'))
            mini_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.value = tk.Checkbutton(mini_frame, variable=self.var, onvalue=1, offvalue=0, command=self._cb_int,
                                        bg=colors.get('background_default'))
            self.value.pack(side=tk.LEFT)
        elif isinstance(arg.choices, (list, tuple)) and len(arg.choices) > 0:
            self.var = tk.StringVar(value=value)
            self.var.trace_add('write', self._cb_choice)
            self.value = ttk.OptionMenu(self, self.var, value, *arg.choices)
            self.value.pack(side=tk.RIGHT, fill=tk.X, expand=1)
        else:
            self.var = tk.StringVar(value=value)
            self.var.trace_add("write", self._cb_str)
            self.value = tk.Entry(self, textvariable=self.var, bg=colors.get('background_default'))
            self.value.pack(side=tk.RIGHT, fill=tk.X, expand=1)

        # add tooltips
        text = "%s\n\ndefault: %s" % (self.arg.help, str(self.arg.default))
        CreateToolTip(self.label, text=text, wraplength=sizes.get('wrap_tooltip'))
        self.value_tooltip = CreateToolTip(self.value, text=text, wraplength=sizes.get('wrap_tooltip'))

    def _cb_str(self, *_, **__):
        if self._config_enabled:
            self.interactive.set_arg_value(self.node.name, self.arg.name, self.var.get())
        return True

    def _cb_choice(self, *_, **__):
        if self._config_enabled:
            self.interactive.set_arg_value(self.node.name, self.arg.name, self.var.get())
            maybe_add_cls_tooltip(self.var.get(), tooltip=self.value_tooltip)
        return True

    def _cb_int(self):
        if self._config_enabled:
            self.interactive.set_arg_value(self.node.name, self.arg.name, "True" if self.var.get() else "False")
        return True

    def _update_content(self):
        value = self.arg.apply_rules(self.node.get_arg_value(self.arg.name))
        if isinstance(self.value, tk.Entry):
            self.value.delete(0, tk.END)
            self.value.insert(tk.END, value)
        if isinstance(self.value, ttk.OptionMenu):
            self.var.set(value)
        if isinstance(self.value, tk.Checkbutton):
            if value:
                self.value.select()
            else:
                self.value.deselect()

    def _highlight_content(self, s: str) -> int:
        if s in self.arg.name.lower() or s in str(self.var.get()).lower():
            self.config(bg=colors.get('background_highlight'))
            return 1
        self.config(bg=colors.get('background_default'))
        return 0


class ArgsFrame(ToggledFrame, UpdatableFrame):
    """
    several one key: value pairs
    """

    def __init__(self, master, interactive: Interactive, node: GuiArgsTreeNode, *args, **kwargs):
        args_ = list(node.yield_args())
        super().__init__(master, *args, **kwargs, text="%d Arguments" % len(args_))
        self.entries = []

        for arg, _ in args_:
            entry = ArgsEntryFrame(self.sub_frame, interactive, node, arg)
            entry.pack(side=tk.TOP, fill=tk.X, expand=1)
            self.entries.append(entry)

    def _highlight_content(self, s: str) -> int:
        if s in self.text.lower():
            self.config(bg=colors.get('background_highlight'))
            return 1
        self.config(bg=colors.get('background_default'))
        return 0


class TkMetaWidget(ToggledFrame, UpdatableFrame):
    """
    wrapping several objects (trainer, data, ...) by cls_<name>
    """

    def __init__(self, master, interactive: Interactive, node: GuiArgsTreeNode, meta: MetaArgument):
        self._config_enabled = False
        super().__init__(master, borderwidth=sizes.get('borderwidth'),
                         text="%s (allowed: %s)" % (meta.argument.name, meta.limit_str()),
                         relief=None, show=True)
        self.interactive = interactive
        self.node = node
        self.meta = meta
        self.children_widgets = []

        self.str_add = '+'
        self.str_rem = '-'

        area_bottom = tk.Frame(master=self.sub_frame, bg=colors.get('background_default'))
        area_bottom.pack(side=tk.BOTTOM, expand=1, fill=tk.X)

        self.var_rem_str = tk.StringVar(value=self.str_rem)
        self.var_rem_str.trace_add('write', self._cb_rem_str)
        self.rem_menu_button = tk.Menubutton(self.title_frame, text=self.str_rem,
                                             borderwidth=1, relief="raised",
                                             indicatoron=True)
        self.rem_menu = tk.Menu(self.rem_menu_button, tearoff=False)
        self.rem_menu_button.configure(menu=self.rem_menu)
        self.rem_menu_button.pack(side=tk.RIGHT)

        self.var_add_str = tk.StringVar(value=self.str_add)
        self.var_add_str.trace_add('write', self._cb_add_str)
        self.add_menu_button = tk.Menubutton(self.title_frame, text=self.str_add,
                                             borderwidth=1, relief="raised",
                                             indicatoron=True)
        self.add_menu = tk.Menu(self.add_menu_button, tearoff=False)
        self.add_menu_button.configure(menu=self.add_menu)
        self.add_menu_button.pack(side=tk.RIGHT)

    def _set_add_options(self) -> int:
        """ add items to the add-child button """
        self.add_menu.delete(0, tk.END)
        added = 0
        for name in self.meta.get_remaining_options(self.get_child_names()):
            self.add_menu.add_radiobutton(label=name, variable=self.var_add_str)
            added += 1
        return added

    def _set_rem_options(self):
        """ add items to the remove-child button """
        self.rem_menu.delete(0, tk.END)
        for name in self.get_child_names():
            self.rem_menu.add_radiobutton(label=name, variable=self.var_rem_str)

    def get_child_names(self) -> [str]:
        return [n.name for n in self.node.children[self.meta.argument.name]]

    def _cb_add_str(self, *_, **__):
        if self._config_enabled:
            try:
                self.interactive.add_meta_value(self.node.name, self.meta.argument.name, self.var_add_str.get())
            except Exception as e:
                LoggerManager().get_logger().error(str(e), exc_info=e)
                tkm.showwarning(message=str(e))
            self.var_add_str.set(self.str_add)
        self.update_content()
        self.update()

    def _cb_rem_str(self, *_, **__):
        if self._config_enabled:
            splits = self.var_rem_str.get().split('#')
            idx = 0 if len(splits) < 2 else int(splits[1])
            try:
                self.interactive.remove_meta_index(self.node.name, self.meta.argument.name, idx)
            except Exception as e:
                LoggerManager().get_logger().error(str(e), exc_info=e)
                tkm.showwarning(message=str(e))
        self.update_content()
        self.update()

    def on_add_child(self, child: GuiArgsTreeNode):
        tk_child = TkNodeWidget(master=self.sub_frame, interactive=self.interactive, node=child)
        tk_child.pack(side=tk.TOP, expand=1, fill=tk.BOTH)
        tk_child.enable_config_changes(True)
        self.children_widgets.append(tk_child)
        self._update_content()
        self.update()

    def on_delete_child(self, child: GuiArgsTreeNode, index: int):
        tk_child = self.children_widgets.pop(index)
        assert tk_child.node == child
        tk_child.destroy()
        self._update_content()
        self.update()

    def _update_content(self):
        # correctness color
        a1, a2 = self.meta.is_allowed_num(len(self.children_widgets))
        if a1 and a2:
            self.config(bg=colors.get('meta_count_correct'))
        elif self.meta.is_optional():
            self.config(bg=colors.get('meta_count_unclear'))
        else:
            self.config(bg=colors.get('meta_count_wrong'))
        # may add a child without problems, show the add button
        if self._set_add_options() > 0:
            self.add_menu_button.pack(side=tk.RIGHT)
        else:
            self.add_menu_button.forget()
        # nothing to remove, hide the remove button
        if len(self.children_widgets) > 0:
            self.rem_menu_button.pack(side=tk.RIGHT)
            self._set_rem_options()
        else:
            self.rem_menu_button.forget()

    def _highlight_content(self, s: str) -> int:
        if s in self.text.lower():
            self.config(bg=colors.get('background_highlight'))
            self._was_highlighted = True
            return 1
        if self._was_highlighted:
            self._update_content()
            return 0
        self.config(bg=colors.get('background_default'))
        return 0

    def reset(self):
        for child in self.children_widgets:
            child.destroy()
        self.children_widgets.clear()
        self._update_content()
        self.update()


class TkNodeWidget(UpdatableFrame, EventHook):
    """
    one object (trainer, data, ...)
    """

    def __init__(self, master, interactive: Interactive, node: GuiArgsTreeNode):
        super().__init__(master, borderwidth=sizes.get('borderwidth'))
        self.interactive = interactive
        self.node = node
        self.node.add_hook(self)
        self.may_delete = self.node is not self.interactive.root
        self.children_widgets = {}

        area_head = tk.Frame(master=self, bg=colors.get('background_default'))
        self.label = tk.Label(area_head, text=self.node.name, bg=colors.get('background_default'))
        self.label.pack(side=tk.LEFT)
        maybe_add_cls_tooltip(self.node.name, label=self.label)
        area_head.pack(side=tk.TOP, expand=1, fill=tk.X)

        area_content = tk.Frame(master=self, bg=colors.get('background_default'))
        area_content.pack(side=tk.BOTTOM, expand=1, fill=tk.BOTH)

        area_left = tk.Frame(master=area_content, bg=colors.get('background_default'))
        area_left.pack(side=tk.LEFT, expand=0, fill=tk.Y, padx=8)

        area_right = tk.Frame(master=area_content, bg=colors.get('background_default'))
        area_right.pack(side=tk.RIGHT, expand=1, fill=tk.BOTH)

        if len(list(self.node.get_arg_values())) > 0:
            self.area_args = ArgsFrame(master=area_right, interactive=interactive, node=node, relief=None, show=True)
            self.area_args.pack(side=tk.TOP, expand=1, fill=tk.BOTH)
        else:
            self.area_args = None

        area_meta = tk.Frame(master=area_right, bg=colors.get('background_default'))
        area_meta.pack(side=tk.BOTTOM, expand=1, fill=tk.BOTH)

        for meta in self.node.args_cls.meta_args_to_add():
            self.children_widgets[meta.argument.name] = TkMetaWidget(area_meta, interactive, node, meta)
            self.children_widgets[meta.argument.name].pack(side=tk.TOP, expand=1, fill=tk.X)

    def _update_content(self):
        self.label.config(text=self.node.name)

    def reset(self):
        self.node.reset()
        for child in self.children_widgets.values():
            child.reset()

    def on_add_child(self, node: GuiArgsTreeNode, child: GuiArgsTreeNode, meta: MetaArgument):
        assert node == self.node
        self.children_widgets[meta.argument.name].on_add_child(child)

    def on_update_indices(self, node: GuiArgsTreeNode, meta: MetaArgument):
        assert node == self.node
        self.update_content()
        self.update()

    def on_delete(self, node: GuiArgsTreeNode):
        assert node == self.node

    def on_delete_child(self, node: GuiArgsTreeNode, child: GuiArgsTreeNode, meta_name: str, index: int):
        assert node == self.node
        self.children_widgets[meta_name].on_delete_child(child, index)

    def _highlight_content(self, s: str) -> int:
        if s in self.node.name.lower():
            self.config(bg=colors.get('background_highlight'))
            return 1
        self.config(bg=colors.get('background_default'))
        return 0


class TkArgsGui(UpdatableFrame, EventHook):
    # very useful for scrollbars: https://gist.github.com/mp035/9f2027c3ef9172264532fcd6262f3b01

    def __init__(self, master):
        super().__init__(master)
        self.master.title("UniNAS Args GUI")
        self.master = master

        # logic
        self.interactive = Interactive()
        self._save_path = None
        self._init_dir = standard_paths.get('path_tmp')

        # menu
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        menu.add_command(label="Load", command=self.load_config)
        menu.add_command(label="Load Add", command=self.load_add_config)
        menu.add_command(label="Save", command=self.save_config)
        menu.add_command(label="Save As", command=self.save_as_config)
        menu.add_command(label="Visualize", command=self.visualize)
        menu.add_command(label="Run", command=self.run)
        menu.add_separator()
        menu.add_command(label="Reset", command=self.reset)
        menu.add_command(label="Exit", command=self.exit)

        # search bar to highlight text
        main_frame = tk.Frame(self, bg=colors.get('background_default'))
        search_frame = tk.Frame(main_frame, bg=colors.get('background_default'))
        self.label_search = tk.Label(search_frame, text='Search:\t', bg=colors.get('background_default'))
        self.label_search.pack(side=tk.LEFT)
        content_frame = tk.Frame(main_frame, bg=colors.get('background_default'))

        search_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.var_str = tk.StringVar(value="")
        self.var_str.trace_add("write", self._cb_search)
        self.value = tk.Entry(search_frame, textvariable=self.var_str)
        self.value.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # scrollable canvas with content
        self.canvas = tk.Canvas(content_frame, background="#ffffff")
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        self.root_node = TkNodeWidget(master=self.canvas, interactive=self.interactive, node=self.interactive.root)
        self.scrollbar = tk.Scrollbar(content_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.root_node.pack(expand=1, fill=tk.BOTH)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_window = self.canvas.create_window((4, 4), window=self.root_node, anchor=tk.NW)

        self.root_node.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.on_frame_configure(None)

        self.pack(side=tk.TOP, expand=1, fill=tk.BOTH)
        self.update_content()
        self.update()

    def _cb_search(self, *_, **__):
        num_hits = self.highlight_content(self.var_str.get())
        text = ('Search (%d hits):\t' % num_hits) if num_hits > 0 else 'Search:\t'
        self.label_search.config(text=text)
        return True

    def on_frame_configure(self, event):
        # reset the scroll region to encompass the inner frame
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        # reset the canvas window to encompass inner frame when required
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def _on_mousewheel(self, event):
        if event.num == 5 or event.delta == -120:
            self.canvas.yview_scroll(misc.get('scroll_delta'), "units")
        if event.num == 4 or event.delta == 120:
            self.canvas.yview_scroll(-misc.get('scroll_delta'), "units")

    def load_config(self):
        self.load_add_config(add=False)

    def load_add_config(self, add=True):
        path = filedialog.askopenfilename(initialdir=self._init_dir,
                                          initialfile=name_task_config,
                                          title="Select file",
                                          filetypes=(("run config files", "*.run_config"),))
        if isinstance(path, str) and os.path.isfile(path) and len(path) > 0:
            self.enable_config_changes(False)
            if not add:
                self.reset()
            try:
                self.interactive.from_json(path)
            except Exception as e:
                LoggerManager().get_logger().error(str(e), exc_info=e)
                tkm.showwarning(message=str(e))
            self.update_content()
            self.update()

    def save_config(self):
        self.save_as_config(path=self._save_path)

    def save_as_config(self, path=None):
        if path is None:
            path = filedialog.asksaveasfilename(initialdir=self._init_dir,
                                                initialfile=name_task_config,
                                                title="Select file",
                                                filetypes=(("run config files", "*.run_config"),))
        if isinstance(path, str) and not os.path.isdir(path) and len(path) > 0:
            try:
                self.interactive.to_json(path)
            except Exception as e:
                LoggerManager().get_logger().error(str(e), exc_info=e)
                tkm.showwarning(message=str(e))
            self._init_dir = os.path.dirname(path)
            self._save_path = path

    def visualize(self):
        visualize_args_tree(self.interactive.root).view(filename="args_tree",
                                                        directory=replace_standard_paths("{path_tmp}"),
                                                        quiet_view=True,
                                                        cleanup=True)

    def run(self):
        try:
            path = replace_standard_paths('{path_tmp}/tmp.run_config')
            self.interactive.to_json(path)
            Main.new_task(path).run()
        except Exception as e:
            LoggerManager().get_logger().error(str(e), exc_info=e)
            tkm.showwarning(message=str(e))

    def reset(self):
        self.root_node.reset()
        self.update_content()
        self.update()

    def exit(self):
        self.master.destroy()

    def _update_content(self):
        self.enable_config_changes(True)


def exp():
    tk_gui = TkArgsGui(master=tk.Tk())
    tk_gui.mainloop()


if __name__ == '__main__':
    exp()
