import sys
import os
import argparse
import matplotlib.pyplot as plt
from typing import Union, List
from uninas.tasks.abstract import AbstractTask
from uninas.utils.args import arg_list_from_json, ArgsInterface, MetaArgument, Argument, ArgsTreeNode
from uninas.builder import Builder
from uninas.register import Register

cla_type = Union[str, List, None]

# prevent tensorflow spam in some situations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only error/fatal


class Main(ArgsInterface):

    @classmethod
    def init(cls):
        """ make sure everything is registered """
        Builder()

    @classmethod
    def list_all_arguments(cls):
        """ list all arguments of all classes that expose arguments """

        cls.init()
        all_to_list = [cls] + [item.value for item in Register.all.values()]
        arg_str = '  {name:<30}{type:<20}{default:<35}{help:<80}{choices}'

        def maybe_print_args(name: str, arguments: [Argument]) -> bool:
            if len(arguments) > 0:
                print('\n%s' % name)
                for a in arguments:
                    choices = ''
                    if isinstance(a.choices, (list, tuple)):
                        choices = [('"%s"' % c) if isinstance(c, str) else str(c) for c in a.choices]
                        choices = 'choices=[%s]' % ', '.join(choices)
                    elif isinstance(a.registered, (list, tuple)):
                        choices = [('"%s"' % c) if isinstance(c, str) else str(c) for c in a.registered]
                        choices = 'meta=[%s]' % ', '.join(choices)
                    print(arg_str.format(**{
                        'name': a.name,
                        'type': str(a.type),
                        'default': str(a.default),
                        'help': a.help,
                        'choices': choices,
                    }))
                return True
            return False

        print('\n', '-'*140,
              '\n', 'these meta arguments influence which classes+arguments will be dynamically added:',
              '\n', '(the classes may have further arguments, listed below)',
              '\n', '-'*140)
        for v in all_to_list:
            if isinstance(v, type) and issubclass(v, ArgsInterface):
                args = v.meta_args_to_add()
                maybe_print_args(v.__name__, [a.argument for a in args])

        print('\n', '-'*140,
              '\n', 'all classes that have arguments:',
              '\n', '-'*140)
        no_args = []
        for v in all_to_list:
            if isinstance(v, type) and issubclass(v, ArgsInterface):
                args = v.args_to_add()
                if not maybe_print_args(v.__name__, args):
                    no_args.append(v)

        print('\n', '-'*140,
              '\n', 'classes that do not define arguments:',
              '\n', '-'*140,
              '\n')
        for v in no_args:
            print(v.__name__)

    @classmethod
    def new_task(cls, cla: cla_type = None, args_changes: dict = None, raise_unparsed=True) -> AbstractTask:
        """
        :param cla:
            str: path to run_config file(s), (separated by commas), overrules other command line args if this exists
            list: specified command line arguments, overrules default system arguments
            None: use the system arguments
        :param args_changes: optional dictionary of changes to the command line arguments
        :param raise_unparsed: raise an exception if there are unparsed arguments left
        :return: new task as defined by the command line arguments
        """
        print("Creating new task")
        cls.init()

        # reset any plotting
        plt.clf()
        plt.cla()

        # from config file?
        if isinstance(cla, str):
            cla = arg_list_from_json(cla)
        print('-'*50)

        # get arguments, insert args_changes
        args_list = sys.argv[1:] if cla is None else cla
        args_changes = args_changes if args_changes is not None else {}
        for k, v in args_changes.items():
            cla.append('--%s=%s' % (k, v))

        parser = argparse.ArgumentParser(description='UniNAS Project')
        node = ArgsTreeNode(Main)
        node.build_from_args(args_list, parser)
        args, wildcards, failed_args, descriptions = node.parse(args_list, parser, raise_unparsed=raise_unparsed)

        # note failed wildcards
        if len(failed_args) > 0:
            print('-'*50)
            print('Failed replacements for argparse:')
            print(', '.join(failed_args))

        # list all wildcards for convenience
        print('-'*50)
        print('Wildcard replacements for argparse:')
        for k, v in wildcards.items():
            print('\t{:<25} ->  {}'.format('{%s}' % k, v))
        print('-'*50)

        # clean up, create and return the task
        Argument.reset_cached()
        cls_task = cls._parsed_meta_argument(Register.tasks, 'cls_task', args, index=None)
        print('Starting %s!' % cls_task.__name__)
        return cls_task(args, wildcards, descriptions=descriptions)

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + [
            MetaArgument('cls_task', Register.tasks, help_name='task', allowed_num=1),
        ]


if __name__ == '__main__':
    Main.list_all_arguments()
