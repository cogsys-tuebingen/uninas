"""
initializing weights of a network
"""

from uninas.modules.modules.abstract import AbstractModule
from uninas.utils.args import ArgsInterface, Namespace
from uninas.utils.loggers.python import logging, LoggerManager


class AbstractInitializer(ArgsInterface):
    def __init__(self, args: Namespace, index=None):
        ArgsInterface.__init__(self)

    def initialize_weights(self, net: AbstractModule):
        logger = LoggerManager().get_logger()
        logger.info('Initializing: %s' % self.__class__.__name__)
        self._initialize_weights(net, logger)

    def _initialize_weights(self, net: AbstractModule, logger: logging.Logger):
        raise NotImplementedError
