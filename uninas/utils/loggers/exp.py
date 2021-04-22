
from pytorch_lightning.loggers.base import LightningLoggerBase, LoggerCollection
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger

from uninas.utils.args import ArgsInterface, Argument, Namespace
from uninas.register import Register


class AbstractExpLogger(ArgsInterface):
    @classmethod
    def from_args(cls, save_dir: str, args: Namespace, index: int = None, version=0) -> LightningLoggerBase:
        parsed = cls._all_parsed_arguments(args, index=index)
        return cls.get_logger(save_dir, version=version, **parsed)

    @classmethod
    def collection(cls, save_dir: str, args: Namespace, cls_loggers: [], version=0) -> LoggerCollection:
        loggers = []
        for i, cls_logger in enumerate(cls_loggers):
            loggers.append(cls_logger.from_args(save_dir, args, index=i, version=version))
        return LoggerCollection(loggers)

    @classmethod
    def get_logger(cls, save_dir: str, version=0, **__) -> LightningLoggerBase:
        raise NotImplementedError


@Register.exp_logger()
class TensorBoardExpLogger(AbstractExpLogger):
    """
    log everything to TensorBoard
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('log_graph', default='True', type=str, help='whether to log the graph structure', is_bool=True),
        ]

    @classmethod
    def get_logger(cls, save_dir: str, version=0, log_graph=True, **__) -> LightningLoggerBase:
        return TensorBoardLogger(save_dir=save_dir, version=version, log_graph=log_graph)


@Register.exp_logger()
class WandbExpLogger(AbstractExpLogger):
    """
    log everything to Wandb
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('offline', default='False', type=str, help='Run offline (data can be streamed later to wandb servers)', is_bool=True),
            Argument('anonymous', default='False', type=str, help='Enables or explicitly disables anonymous logging', is_bool=True),
            Argument('project', default="project", type=str, help='The name of the project to which this run will belong'),
            Argument('log_model', default='False', type=str, help='Save checkpoints in wandb dir to upload on W&B servers', is_bool=True),
        ]

    @classmethod
    def get_logger(cls, save_dir: str, version=0, offline=False, anonymous=False, project=None, log_model=True, **__) -> LightningLoggerBase:
        return WandbLogger(save_dir=save_dir, version=str(version), offline=offline, anonymous=anonymous, project=project, log_model=log_model)


@Register.exp_logger()
class CsvExpLogger(AbstractExpLogger):
    """
    log everything to a CSV file
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('name', default=None, type=str, help='optional name'),
        ]

    @classmethod
    def get_logger(cls, save_dir: str, version=0, name=None, **__) -> LightningLoggerBase:
        return CSVLogger(save_dir=save_dir, version=str(version), name=name)
