"""
networks loaded from a config file
"""


import os
from uninas.methods.strategy_manager import StrategyManager
from uninas.methods.random import RandomChoiceStrategy
from uninas.models.networks.uninas.abstract import AbstractUninasNetwork
from uninas.models.networks.uninas.search import SearchUninasNetwork
from uninas.utils.loggers.python import LoggerManager
from uninas.utils.args import MetaArgument, Argument, Namespace, find_in_args_list
from uninas.utils.shape import Shape, ShapeList
from uninas.utils.misc import split
from uninas.register import Register
from uninas.builder import Builder


@Register.network()
class RetrainFromSearchUninasNetwork(AbstractUninasNetwork):
    """
    create a search network, extract the specific architecture as a stand-alone network
    """

    def __init__(self, search_config_path: str, gene: str, *args, **kwargs):
        super().__init__(model_name=gene, net=None, *args, **kwargs)
        self._add_to_kwargs(gene=gene, search_config_path=search_config_path)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('search_config_path', type=str, default='not_set', help='path to the search dir', is_path=True),
            Argument('gene', default='random', type=str, help='"random" or [int] gene of weights in super-net'),
        ]

    @classmethod
    def from_args(cls, args: Namespace, index=None, **_) -> 'RetrainFromSearchUninasNetwork':
        """
        :param args: global argparse namespace
        :param index: index for the args
        """
        all_parsed = cls._all_parsed_arguments(args, index=index)
        return cls(**all_parsed)

    def _build2(self, s_in: Shape, s_out: Shape) -> ShapeList:
        """ build the network """

        # find the search config
        if not os.path.isfile(self.search_config_path):
            self.search_config_path = Builder.find_net_config_path(self.search_config_path, pattern='search')

        # create a temporary search strategy
        tmp_s = RandomChoiceStrategy(max_epochs=1, name='__tmp__')
        sm = StrategyManager()
        assert len(sm.get_strategies_list()) == 0, "can not load when there already is a search network"
        sm.add_strategy(tmp_s)
        sm.set_fixed_strategy_name('__tmp__')

        # create a search network
        search_net = Register.builder.load_from_config(self.search_config_path)
        assert isinstance(search_net, SearchUninasNetwork)
        search_net.build(s_in, s_out)
        search_net.set_forward_strategy(False)

        # set the architecture, get the config
        req_gene = ""
        if self.gene == 'random':
            search_net.forward_strategy()
            gene = sm.get_all_finalized_indices(unique=True, flat=True)
            self.model_name = "random(%s)" % str(gene)
            req_gene = " (%s)" % self.gene
        else:
            gene = split(self.gene, int)
        l0, l1 = len(sm.get_all_finalized_indices(unique=True)), len(gene)
        assert l0 == l1, "number of unique choices in the network (%d) must match length of the gene (%d)" % (l0, l1)
        search_net.forward_strategy(fixed_arc=gene)
        config = search_net.config(finalize=True)

        # clean up
        sm.delete_strategy('__tmp__')
        del sm
        del search_net

        # build the actually used finalized network
        LoggerManager().get_logger().info("Extracting architecture %s%s from the super-network" % (gene, req_gene))
        self.net = Register.builder.from_config(config)
        return self.net.build(s_in, s_out)

    def config(self, **_) -> dict:
        return self.net.config(**_)


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
    def from_args(cls, args: Namespace, index=None) -> 'RetrainUninasNetwork':
        """
        :param args: global argparse namespace
        :param index: index for the args
        """
        all_parsed = cls._all_parsed_arguments(args, index=index)
        config_path = all_parsed.pop('config_path')
        config_path = Builder.find_net_config_path(config_path)
        net = Register.builder.load_from_config(config_path)
        return cls(model_name=Builder.net_config_name(config_path), net=net, **all_parsed)

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
    def from_args(cls, args: Namespace, index=None) -> 'RetrainInsertConfigUninasNetwork':
        """
        :param args: global argparse namespace
        :param index: argument index
        """
        all_parsed = cls._all_parsed_arguments(args, index=index)
        config_path = Builder.find_net_config_path(all_parsed.pop('config_path'))
        net = cls._parsed_meta_argument(Register.network_bodies, 'cls_network_body', args, index=index)
        net = net.search_network_from_args(args, index=index)
        net.add_cells_from_config(Register.builder.load_config(config_path))
        return cls(model_name=Builder.net_config_name(config_path), net=net, **all_parsed)

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
        config_path = Builder.find_net_config_path(config_path)

        # extract used classes from the network config file, add them to the current task config if missing
        used_classes = Builder().find_classes_in_config(config_path)
        network_name = used_classes['cls_network_body'][0]
        cls_network_body = Register.network_bodies.get(network_name)
        optional_meta = [m.argument.name for m in cls_network_body.meta_args_to_add() if m.optional_for_loading]
        print('\tbuilding a new net (config_only=False), added missing args from the network config file')
        for cls_n in ['cls_network_body'] + optional_meta:
            cls_c = find_in_args_list(args_list, [cls_n])
            if cls_c is None or len(cls_c) == 0:
                cls_v = ', '.join(used_classes[cls_n])
                print('\t  %s -> %s' % (cls_n, cls_v))
                args_list.append('--%s=%s' % (cls_n, cls_v))
