from uninas.training.initializers.abstract import AbstractInitializer
from uninas.networks.self.abstract import AbstractUninasNetwork
from uninas.model.modules.abstract import AbstractModule
from uninas.training.callbacks.checkpoint import CheckpointCallback
from uninas.utils.args import Namespace, Argument
from uninas.utils.misc import split
from uninas.utils.loggers.python import logging
from uninas.register import Register


@Register.initializer()
class LoadWeightsInitializer(AbstractInitializer):
    """
    Simply load weights from a checkpoint

    if the checkpoint is of a s1 network and only certain paths are available in the current (s3) network,
    use the given 'gene' to select the weights and match them by shape
    """

    def __init__(self, args: Namespace, index=None):
        super().__init__(args, index)
        self.path, gene, self.strict = self._parsed_arguments(['path', 'gene', 'strict'], args, index=index)
        self.gene = split(gene, int)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('path', default='{path_tmp}/s1/', type=str, help='path', is_path=True),
            Argument('gene', default='', type=str, help='gene of weights in super-net'),
            Argument('strict', default='True', type=str, help='strict loading', is_bool=True),
        ]

    def _initialize_weights(self, net: AbstractModule, logger: logging.Logger):
        assert isinstance(net, AbstractUninasNetwork), "This initializer will not work with external networks!"
        checkpoint = CheckpointCallback.load_last_checkpoint(self.path)
        state_dict = checkpoint.get('state_dict')
        added_state = checkpoint.get('net_add_state', None)

        # figure out correct weights in supernet checkpoint
        if len(self.gene) > 0 and added_state is not None:
            logger.info(' > loading weights of gene %s from checkpoint "%s"' % (str(self.gene), self.path))
            target_dict = net.state_dict()
            target_names = list(target_dict.keys())
            new_dict = {}
            format_str = ' > {:24}{:<68}{:<60}{:24}'
            # add all stem and head weights, they are at the front of the dict and have pretty much the same name
            logger.debug(format_str.format('shape in checkpoint', 'name in checkpoint', 'name in net', 'shape in net'))
            for k, v in state_dict.items():
                if '.stem.' in k or '.heads.' in k:
                    tn = target_names.pop(0)
                    ts = target_dict[tn].shape
                    logger.debug(format_str.format(str(list(v.shape)), k, tn, str(list(ts))))
                    n = k.replace('net.', '', 1)
                    assert n == tn
                    new_dict[n] = v
            # add all cell weights, can generally not compare names, only shapes
            for i, cell in enumerate(added_state.get('cells', list())):
                choice = cell[self.gene[i]]
                for name, shape, trainable in choice:
                    tn = target_names.pop(0)
                    ts = target_dict[tn].shape
                    logger.debug(format_str.format(str(list(shape)), name, tn, str(list(ts))))
                    assert shape == ts, 'Mismatching shapes for "%s" and "%s", is the gene correct?' % (name, tn)
                    new_dict[tn] = state_dict[name]
            net.load_state_dict(new_dict, strict=self.strict)

        # simply load
        else:
            logger.info(' > simply loading state_dict')
            net.load_state_dict(state_dict, strict=self.strict)
