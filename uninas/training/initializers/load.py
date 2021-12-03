from uninas.training.initializers.abstract import AbstractInitializer
from uninas.models.networks.uninas.abstract import AbstractUninasNetwork
from uninas.models.networks.uninas.search import SearchUninasNetwork
from uninas.modules.modules.abstract import AbstractModule
from uninas.training.callbacks.checkpoint import CheckpointCallback
from uninas.methods.strategy_manager import StrategyManager
from uninas.methods.random import RandomChoiceStrategy
from uninas.utils.args import Namespace, Argument
from uninas.utils.misc import split
from uninas.utils.loggers.python import logging, log_in_columns, log_headline
from uninas.register import Register
from uninas.builder import Builder


@Register.initializer()
class LoadWeightsInitializer(AbstractInitializer):
    """
    Simply load weights from a checkpoint

    if the checkpoint is of a s1 network and only certain paths are available in the current (s3) network,
    try to
        1) build the search network of identical size (by input/output shapes)
        2) set the architecture accordingly, track which weights have been used in the forward pass
        3) match weights by name + shape
        4) load the corresponding weights from the super-network checkpoint
    """

    def __init__(self, args: Namespace, index=None):
        super().__init__(args, index)
        self.path, gene, self.strict = self._parsed_arguments(['path', 'gene', 'strict'], args, index=index)
        self.gene = split(gene, int)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('path', default='{path_tmp}/s1/', type=str, help='path to a search save dir', is_path=True),
            Argument('gene', default='', type=str, help='gene of weights in super-net'),
            Argument('strict', default='True', type=str, help='strict loading', is_bool=True),
        ]

    def _initialize_weights(self, net: AbstractModule, logger: logging.Logger):
        assert isinstance(net, AbstractUninasNetwork), "This initializer will not work with external networks!"
        search_config = Builder.find_net_config_path(self.path, pattern='search')

        checkpoint = CheckpointCallback.load_last_checkpoint(self.path)
        state_dict = checkpoint.get('state_dict')

        # figure out correct weights in super-network checkpoint
        if len(self.gene) > 0:
            log_headline(logger, "tmp network to track used params", target_len=80)
            sm = StrategyManager()
            tmp_s = RandomChoiceStrategy(max_epochs=1, name='__tmp__')
            assert len(sm.get_strategies_list()) == 0, "can not load when there already is a search network"
            sm.add_strategy(tmp_s)
            sm.set_fixed_strategy_name('__tmp__')

            search_net = Builder().load_from_config(search_config)
            assert isinstance(search_net, SearchUninasNetwork)
            s_in, s_out = net.get_shape_in(), net.get_shape_out()
            search_net.build(s_in, s_out[0])
            search_net.set_forward_strategy(False)
            search_net.forward_strategy(fixed_arc=self.gene)
            tracker = search_net.track_used_params(s_in.random_tensor(batch_size=2))
            # tracker.print()

            logger.info(' > loading weights of gene %s from checkpoint "%s"' % (str(self.gene), self.path))
            target_dict = net.state_dict()
            target_names = list(target_dict.keys())
            new_dict = {}

            # add all stem and head weights, they are at the front of the dict and have pretty much the same name
            log_columns = [('shape in checkpoint', 'name in checkpoint', 'name in network', 'shape in network')]
            for k, v in state_dict.items():
                if '.stem.' in k or '.heads.' in k:
                    tn = target_names.pop(0)
                    ts = target_dict[tn].shape
                    log_columns.append((str(list(v.shape)), k, tn, str(list(ts))))
                    n = k.replace('net.', '', 1)
                    assert n == tn
                    new_dict[n] = v

            # add all cell weights, can generally not compare names, only shapes
            for i, tracker_cell_entry in enumerate(tracker.get_cells()):
                for entry in tracker_cell_entry.get_pareto_best():
                    tn = target_names.pop(0)
                    ts = target_dict[tn].shape
                    log_columns.append((str(list(entry.shape)), entry.name, tn, str(list(ts))))
                    assert entry.shape == ts,\
                        'Mismatching shapes for "%s" and "%s", is the gene correct?' % (entry.name, tn)
                    new_dict[tn] = state_dict[entry.name]

            # log matches, load
            log_in_columns(logger, log_columns, add_bullets=True, num_headers=1)
            net.load_state_dict(new_dict, strict=self.strict)

            # clean up
            del search_net
            sm.delete_strategy('__tmp__')
            del sm

        # simply load
        else:
            logger.info(' > simply loading state_dict')
            net.load_state_dict(state_dict, strict=self.strict)
