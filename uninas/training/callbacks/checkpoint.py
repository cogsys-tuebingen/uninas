import os
import time
import glob
from collections import defaultdict
from typing import Union
import torch
import torch.nn as nn
from uninas.methods.abstract_method import AbstractMethod
from uninas.training.trainer.abstract import AbstractTrainerFunctions
from uninas.training.callbacks.abstract import AbstractCallback, EpochInfo
from uninas.utils.torch.misc import itemize
from uninas.utils.paths import replace_standard_paths, maybe_download, FileType
from uninas.utils.args import Argument
from uninas.utils.loggers.python import LoggerManager
from uninas.utils.types import CheckpointType
from uninas.register import Register


@Register.training_callback()
class CheckpointCallback(AbstractCallback):
    """
    Always keep the most recent checkpoint and the top n best ones,
    determined by comparing the values in the logged stats (loss, metrics, ...) using 'key'
    """

    def __init__(self, save_dir: str, index: int, save_clone=True, top_n=2, key='test/loss',
                 minimize_key=True, save_type=CheckpointType.STATE.value):
        """
        :param save_dir: main dir where to save
        :param index: index of this callback
        :param save_clone: save the clone-model weights if available
        :param top_n: keep top n best saves
        :param key: key to rank saves
        :param minimize_key: whether a smaller value is better
        :param save_type: save either:
            state: the current training state, includes model weights, but needs a fitting model to load to
            model: the stand-alone model state that can be used to easily recreate it, but no training info
        """
        super().__init__(save_dir, index)
        self._data = {
            'meta': {},
            'info': defaultdict(EpochInfo),
            'best': [],
            'last': -1,
        }
        # maybe there is saved info
        if os.path.isfile(self._meta_path()):
            self._data.update(torch.load(self._meta_path()))
        # some new info is kept regardless
        self._data.update({
            'save_clone': save_clone,
            'top_n': top_n,
            'key': key,
            'minimize_key': minimize_key,
            'save_type': save_type,
        })
        self._default = 9999999 if self._data['minimize_key'] else -9999999

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('save_clone', default="True", type=str, is_bool=True,
                     help="save the weights of the first clone-model if available, "
                          "otherwise save the trained model's weights"),
            Argument('top_n', default=1, type=int, help='keep top n best saves'),
            Argument('key', default="test/loss", type=str, help='key to rank saves'),
            Argument('minimize_key', default="False", type=str, help='whether a smaller value is better', is_bool=True),
            Argument('save_type', default="state", type=str, choices=CheckpointType.all_types(), help='save either the model state, or the entire model'),
        ]

    def on_train_epoch_end(self, trainer: AbstractTrainerFunctions,
                           pl_module: AbstractMethod,
                           log_dict: dict = None):
        """ Called when the train epoch ends. """
        if trainer.is_rank_zero():
            self.save_track(trainer.choose_method(prefer_clone=self._data.get('save_clone')),
                            log_dict, overwrite=True, update_dict=trainer.get_checkpoint_update_dict())

    def on_validation_epoch_end(self, trainer: AbstractTrainerFunctions,
                                pl_module: AbstractMethod,
                                log_dict: dict = None):
        """ Called when the val epoch ends. """
        if trainer.is_rank_zero():
            self.save_track(pl_module, log_dict, overwrite=False)  # same epoch will not overwrite the train checkpoint

    def on_test_epoch_end(self, trainer: AbstractTrainerFunctions,
                          pl_module: AbstractMethod,
                          log_dict: dict = None):
        """ Called when the test epoch ends. """
        if trainer.is_rank_zero():
            self.save_track(pl_module, log_dict, overwrite=False)  # same epoch will not overwrite the train checkpoint

    def _meta_path(self) -> str:
        return '%s/checkpoints/%d/meta.pt' % (self._save_dir, self._index)

    @classmethod
    def _general_checkpoint_file(cls, save_dir: str) -> str:
        return '%s/checkpoints/checkpoint.pt' % save_dir

    def _relative_checkpoint_name(self, epoch: int, save_type=CheckpointType.STATE.value) -> str:
        return '%s/checkpoints/%d/%s-%d.pt' % ('%s', self._index, save_type, epoch)

    def get_top_n(self, key: str = None) -> [EpochInfo]:
        key = self._data['key'] if key is None else key
        return sorted(self._data['info'], key=lambda d: d.log_dict.get(key, self._default), reverse=self._default < 0)

    def get_top_n_with_paths(self, include_last=True, key: str = None) -> [EpochInfo]:
        key = self._data['key'] if key is None else key
        data = [self._data['info'][i] for i in self._data['best']]
        if include_last and self._data['last'] >= 0:
            info = self._data['info'][self._data['last']]
            if isinstance(info.checkpoint_path, str):
                data.append(info)
        return sorted(data, key=lambda d: d.log_dict.get(key, self._default), reverse=self._default < 0)

    def _auto_clean(self):
        """ remove the worst checkpoints until only the top n remain """
        removed_epochs = []
        top = list(self.get_top_n_with_paths(include_last=False))
        while len(top) > self._data['top_n']:
            data = top.pop(-1)
            if isinstance(data.checkpoint_path, str):
                path = data.checkpoint_path % self._save_dir
                if os.path.isfile(path):
                    os.remove(path)
            data.checkpoint_path = None
            removed_epochs.append(data.epoch)
        for e in removed_epochs:
            self._data['best'].remove(e)

    @classmethod
    def wait_load(cls, file_path: str, td=0.1) -> dict:
        """
        wait until the checkpoint exists, then return it
        """
        while not os.path.isfile(file_path):
            time.sleep(td)
        return torch.load(file_path)

    @classmethod
    def atomic_save(cls, file_path: str, checkpoint: dict):
        """
        atomic saving of a checkpoint
        """
        tmp = file_path + '.part'
        torch.save(checkpoint, tmp)
        os.replace(tmp, file_path)

    @classmethod
    def save(cls, file_path: str, pl_module: AbstractMethod, update_dict: dict = None, save_type=CheckpointType.STATE.value) -> dict:
        """
        save method checkpoint to file, not tracking it, and return the checkpoint
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if CheckpointType.STATE == save_type:
            checkpoint = dict(state_dict=pl_module.state_dict(), __type__=CheckpointType.STATE.value)
            if isinstance(update_dict, dict):
                checkpoint.update(update_dict)
            pl_module.on_save_checkpoint(checkpoint)
        elif CheckpointType.MODEL == save_type:
            checkpoint = pl_module.get_network().get_model_state()
        else:
            raise NotImplementedError("No save routine for save_type = %s" % save_type)
        cls.atomic_save(file_path, checkpoint)
        LoggerManager().get_logger().info('Saved weights to file: %s' % file_path)
        return checkpoint

    @classmethod
    def load(cls, file_path: str, pl_module: Union[AbstractMethod, None] = None,
             ensure_save_type: Union[CheckpointType, str, None] = None) -> dict:
        """
        load method checkpoint from method checkpoint file and return it
        if 'ensure_save_type' is given, raise an exception if the checkpoint does not match it
        if 'pl_module' is given, also have it load the checkpoint
        """
        file_path = replace_standard_paths(file_path)
        logger = LoggerManager().get_logger()
        if os.path.isfile(file_path):
            logger.info('Found checkpoint: %s' % file_path)
            checkpoint = torch.load(file_path)
            if isinstance(pl_module, AbstractMethod):
                checkpoint_type = checkpoint.get('__type__', CheckpointType.STATE.value)
                if ensure_save_type is not None:
                    assert ensure_save_type == checkpoint_type,\
                        "Checkpoint has type %s, but is required to have type %s" % (checkpoint_type, ensure_save_type)
                if CheckpointType.STATE == checkpoint_type:
                    pl_module.load_state_dict(checkpoint['state_dict'])
                    pl_module.on_load_checkpoint(checkpoint)
                    logger.info('Loaded state from file: %s' % file_path)
                elif CheckpointType.MODEL == checkpoint_type:
                    pl_module.get_network().load_state(checkpoint)
                    logger.info('Loaded model from file: %s (does not include info about the training state)' % file_path)
                else:
                    raise NotImplementedError("No load routine for checkpoint type = %s" % checkpoint_type)
            return checkpoint
        else:
            logger.info('Can not load weights, does not exist / not a file: %s' % file_path)
            return {}

    @classmethod
    def load_network(cls, file_path: str, network: nn.Module, num_replacements=1) -> bool:
        """
        load network checkpoint from method checkpoint file
        replace parts of the param names to match the requirements
        """
        checkpoint = cls.load_last_checkpoint(file_path)
        if len(checkpoint) > 0:
            state_dict, net_state_dict = checkpoint.get('state_dict', checkpoint), {}

            # map state dict keys accordingly
            key_mappings = {'net.': ''}
            for key0, v in state_dict.items():
                key1 = key0
                for k0, k1 in key_mappings.items():
                    key1 = key1.replace(k0, k1, num_replacements)
                net_state_dict[key1] = v
            network.load_state_dict(net_state_dict, strict=True)

            LoggerManager().get_logger().info('Loaded weights from file: %s' % file_path)
            return True
        else:
            return False

    @classmethod
    def list_infos(cls, save_dir: str, index=0) -> [EpochInfo]:
        """ list all epoch infos, adapt paths """
        cp = cls(save_dir, index=index)
        lst = cp.get_top_n_with_paths(include_last=True)
        lst = [info for info in lst if isinstance(info.checkpoint_path, str)]
        for info in lst:
            info.checkpoint_path = info.checkpoint_path % cp._save_dir
        return lst

    @classmethod
    def find_last_checkpoint_path(cls, save_dir: str, index=0, try_general_checkpoint=True) -> str:
        """
        attempt finding the checkpoint path in a dir,
        if 'save_dir' is a file, return it
        if there is a general checkpoint and 'try_general_checkpoint', return its path
        otherwise try finding the most recent checkpoint of the CheckpointCallback with index 'index'
        """
        save_dir = replace_standard_paths(save_dir)
        # try as path and general path
        if os.path.isfile(save_dir):
            return save_dir
        if try_general_checkpoint and os.path.isfile(cls._general_checkpoint_file(save_dir)):
            return cls._general_checkpoint_file(save_dir)
        # try by index
        lst = sorted(cls.list_infos(save_dir, index), key=lambda inf: inf.checkpoint_path)
        if len(lst) > 0:
            return lst[-1].checkpoint_path
        # try to find any checkpoint.pt in dir
        for path in glob.glob('%s/**/checkpoint.pt' % save_dir, recursive=True):
            return path
        # failure
        LoggerManager().get_logger().info('Can not find a uninas checkpoint (history) in: %s' % save_dir)
        return ''

    @classmethod
    def find_pretrained_weights_path(cls, path: str, name: str = 'pretrained', raise_missing=True) -> str:
        """
        attempt finding pretrained weights in a dir,
        no matter if checkpoint or external
        """
        maybe_path = maybe_download(path, FileType.WEIGHTS)
        if isinstance(maybe_path, str):
            return maybe_path
        path = replace_standard_paths(path)
        if len(path) == 0 or os.path.isfile(path):
            return path
        # try looking for a checkpoint
        p = cls.find_last_checkpoint_path(path)
        if os.path.isfile(p):
            return p
        # glob any .pt/.pth file with the network name in it
        glob_path = '%s/**/*%s*.pt*' % (path, name)
        paths = glob.glob(glob_path, recursive=True)
        if len(paths) > 0:
            return paths[0]
        # failure
        if raise_missing:
            raise FileNotFoundError('can not find any pretrained weights in "%s" or "%s"' % (path, glob_path))
        return ''

    @classmethod
    def load_last_checkpoint(cls, save_dir: str, pl_module: AbstractMethod = None,
                             index=0, try_general_checkpoint=True) -> dict:
        """
        attempt loading from a dir,
        if 'save_dir' is a file, load it
        if there is a general checkpoint and 'try_general_checkpoint', load it
        otherwise try loading the most recent checkpoint of the CheckpointCallback with index 'index'
        """
        path = cls.find_last_checkpoint_path(save_dir, index, try_general_checkpoint)
        if os.path.isfile(path):
            return cls.load(path, pl_module=pl_module)
        return {}

    def save_track(self, pl_module: AbstractMethod, log_dict: dict = None, overwrite=False, update_dict: dict = None):
        """ possibly save and track the new checkpoint """
        log_dict = {} if log_dict is None else itemize(log_dict)
        epoch = pl_module.current_epoch
        do_save = overwrite
        self._data['info'][epoch].epoch = epoch
        self._data['info'][epoch].log_dict.update(log_dict)

        if epoch != self._data['last']:
            if self._data['last'] >= 0:
                self._data['best'].append(self._data['last'])
            self._data['last'] = epoch
            do_save = True
        if do_save:
            save_type = self._data.get('save_type')
            path = self._relative_checkpoint_name(epoch, save_type=save_type)
            self._data['info'][epoch].checkpoint_path = path
            self.save(path % self._save_dir, pl_module, update_dict=update_dict, save_type=save_type)
            self._auto_clean()
            torch.save(self._data, self._meta_path())
