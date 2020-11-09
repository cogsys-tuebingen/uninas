"""
networks loaded from a config file
"""


import torch
from uninas.networks.self.abstract import AbstractUninasNetwork
from uninas.utils.args import MetaArgument, Argument, Namespace, find_in_args_list
from uninas.utils.paths import find_net_config_path, find_net_config_name
from uninas.register import Register
from uninas.builder import Builder


@Register.network(only_config=True)
class RetrainUninasNetwork(AbstractUninasNetwork):
    """
    load the network topology from a net_config file
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('config_path', type=str, default='not_set', help='path to the config file', is_path=True),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None, **_):
        """
        :param args: global argparse namespace
        :param index: index for the args
        """
        all_parsed = cls._all_parsed_arguments(args, index=index)
        config_path = all_parsed.pop('config_path')
        config_path = find_net_config_path(config_path)
        net = Register.builder.load_from_config(config_path)
        return cls(find_net_config_name(config_path), net, **all_parsed)

    def forward(self, x: torch.Tensor) -> [torch.Tensor]:
        return self.net(x)

    def config(self, **_) -> dict:
        return self.net.config(**_)


@Register.network()
class RetrainInsertConfigUninasNetwork(RetrainUninasNetwork):
    """
    load the network topology from a net_config file and insert it into the used task_config
    this enables changing some details such as cell order
    """

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add() + [
            MetaArgument('cls_network_body', Register.network_bodies, allowed_num=(0, 1), use_index=False,
                         help_name='network body type'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None, **_):
        """
        :param args: global argparse namespace
        :param index: index for the args
        """
        all_parsed = cls._all_parsed_arguments(args, index=index)
        config_path = find_net_config_path(all_parsed.pop('config_path'))
        net = cls._parsed_meta_argument('cls_network_body', args, index=index).search_network_from_args(args)
        net.add_cells_from_config(Register.builder.load_config(config_path))
        return cls(find_net_config_name(config_path), net, **all_parsed)

    @classmethod
    def extend_args(cls, args_list: [str]):
        """
        allow modifying the arguments list before other classes' arguments are dynamically added
        this should be used sparsely, as it is hard to keep track of
        """
        # find last cls_network_body
        super().extend_args(args_list)

        # first find the correct config path, which is in all_args, enable short names (not only full paths)
        config_path = find_in_args_list(args_list, ['{cls_network}.config_path', '%s.config_path' % cls.__name__])

        # extract used classes from the network config file, add them to the current task config if missing
        used_classes = Builder().find_classes_in_config(config_path)
        network_name = used_classes['cls_network_body'][0]
        cls_network_body = Register.get(network_name)
        optional_meta = [m.argument.name for m in cls_network_body.meta_args_to_add() if m.optional_for_loading]
        print('\tbuilding a new net (config_only=False), added missing args from the network config file')
        for cls_n in ['cls_network_body'] + optional_meta:
            cls_c = find_in_args_list(args_list, [cls_n])
            if cls_c is None or len(cls_c) == 0:
                cls_v = ', '.join(used_classes[cls_n])
                print('\t  %s -> %s' % (cls_n, cls_v))
                args_list.append('--%s=%s' % (cls_n, cls_v))
