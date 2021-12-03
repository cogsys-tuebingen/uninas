from typing import Iterable
from uninas.methods.abstract_method import AbstractMethod
from uninas.methods.strategy_manager import StrategyManager
from uninas.modules.mixed.weights import SplitWeightsMixedOp
from uninas.training.trainer.abstract import AbstractTrainerFunctions
from uninas.training.callbacks.abstract import AbstractCallback
from uninas.utils.loggers.python import LoggerManager
from uninas.utils.misc import split
from uninas.utils.args import Argument
from uninas.register import Register


"""
alternate option how this could work:
    - use the extend_args method (should be called before MixedOps are initialized)
      to set a MixedOp depth class attribute
    - MixedOp initializes with full depth from the start, adds all weights to the optimizer,
      and later copies current weights accordingly
however, the current version:
    - is easily backward compatible
    - requires less memory if operations are masked
"""


@Register.training_callback()
class SplitWeightsMixedOpCallback(AbstractCallback):
    """
    Increase the depth of MixedOp for architecture search,
     - this will make copies of the currently used candidates, one copy per available op in the previous op decision
     - depending on the previous decision, another set of weights will be chosen (but otherwise identical operations)
     - this should enable a much better fine tuning
     - has currently no influence on how the operations at each layer are chosen and requires single-path sampling
    """

    def __init__(self, save_dir: str, index: int, milestones: str, pattern: str):
        """
        """
        super().__init__(save_dir, index)
        self.current_depth = 0
        self.milestones = split(milestones, int)
        self.pattern = [p.lower().startswith("t") or p == '1' for p in split(pattern)]
        if len(self.pattern) == 0:
            self.pattern = [True]

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument("milestones", default="100", type=str, help="[int] at which epochs to increase MixedOp depth"),
            Argument("pattern", default="True", type=str, help="[bool] repeated patten when to apply changes"),
        ]

    @classmethod
    def get_mixed_ops(cls, pl_module: AbstractMethod) -> [SplitWeightsMixedOp]:
        if pl_module is None:
            return []
        net = pl_module.get_network()
        return list(net.base_modules_by_condition(lambda m2: isinstance(m2, SplitWeightsMixedOp), recursive=True))

    def _iterate_pattern(self) -> Iterable[bool]:
        while True:
            for p in self.pattern:
                yield p

    def setup(self, trainer: AbstractTrainerFunctions, pl_module: AbstractMethod, stage: str):
        """ Called when fit or test begins """
        # ensure that the model contains MixedOps, and that only single-path weight strategies are used,
        # and that MixedOp can handle the maximum depth
        assert len(self.get_mixed_ops(pl_module)) > 0,\
            "The network contains no %s that could be changed (-> change in primitives)" % SplitWeightsMixedOp.__name__
        assert StrategyManager().is_only_single_path(),\
            "Every architecture strategy must be single-path!"
        assert SplitWeightsMixedOp.max_depth >= len(self.milestones),\
            "The planned depth exceeds the limit set by %s" % SplitWeightsMixedOp.__name__
        # increase initial depth, in case training is continued
        for i in range(pl_module.current_epoch):
            if i in self.milestones:
                self.current_depth += 1
        logger = LoggerManager().get_logger()
        logger.info('%s: init depth=%d, pattern=%s' % (self.__class__.__name__, self.current_depth, str(self.pattern)))
        count, mixed_ops = 0, self.get_mixed_ops(pl_module)
        for mixed_op, apply in zip(mixed_ops, self._iterate_pattern()):
            if apply:
                mixed_op.change_depth(self.current_depth)
                count += 1
        if count > 0 and self.current_depth > 0:
            logger.info('%s: configured %d ops for %s' %
                        (self.__class__.__name__, count, pl_module.__class__.__name__))

    def on_train_epoch_start(self, trainer: AbstractTrainerFunctions,
                             pl_module: AbstractMethod,
                             log_dict: dict = None):
        """ Called when the train epoch begins. """
        for i in range(self.milestones.count(pl_module.current_epoch)):
            self.current_depth += 1
            logger = LoggerManager().get_logger()
            logger.info('%s: depth=%d, pattern=%s' % (self.__class__.__name__, self.current_depth, str(self.pattern)))
            # TODO also apply to network clones (e.g. using EMA weights)
            count, mixed_ops = 0, self.get_mixed_ops(pl_module)
            for mixed_op, apply in zip(mixed_ops, self._iterate_pattern()):
                if apply:
                    mixed_op.change_depth(self.current_depth)
                    count += 1
            if count > 0:
                logger.info('%s: configured %d mixed ops for %s' %
                            (self.__class__.__name__, count, pl_module.__class__.__name__))
        # log
        log_dict_ = {
            self._dict_key("depth"): self.current_depth
        }
        pl_module.log_metrics(log_dict_)
