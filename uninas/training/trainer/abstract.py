from typing import Iterable, Tuple
from torch.optim.optimizer import Optimizer
from uninas.methods.abstract_method import AbstractMethod
from uninas.models.networks.abstract import AbstractNetwork
from uninas.training.clones.abstract import AbstractMethodClone
from uninas.training.optimizers.abstract import WrappedOptimizer
from uninas.training.regularizers.abstract import AbstractRegularizer
from uninas.training.schedulers.abstract import AbstractScheduler


class AbstractTrainerFunctions:
    """
    to avoid cyclic imports with the actual trainer implementations while still ensuring some methods
    (e.g. in callbacks)
    """

    def get_rank(self) -> int:
        return 0

    def is_rank_zero(self) -> bool:
        return self.get_rank() == 0

    def _trigger_callbacks(self, callback_fun: str, *args, **kwargs):
        raise NotImplementedError

    def get_save_dir(self) -> str:
        raise NotImplementedError

    def is_test_run(self) -> bool:
        raise NotImplementedError

    def get_metrics_save_dir(self, model_type='default') -> str:
        return '%s/metrics/%s/' % (self.get_save_dir(), model_type)

    def get_method(self) -> AbstractMethod:
        raise NotImplementedError

    def get_network(self) -> AbstractNetwork:
        return self.get_method().get_network()

    def get_method_clones(self) -> [AbstractMethodClone]:
        """ get the method clones """
        raise NotImplementedError

    def get_regularizers(self) -> [AbstractRegularizer]:
        """ get regularizers """
        return self.get_method().get_regularizers()

    def get_optimizers(self) -> [Optimizer]:
        """ get optimizers """
        raise NotImplementedError

    def get_schedulers(self) -> [AbstractScheduler]:
        """ get schedulers """
        raise NotImplementedError

    def get_optimizer_log_dict(self) -> dict:
        return WrappedOptimizer.get_optimizer_log_dict(self.get_optimizers())

    def choose_method(self, prefer_clone=True) -> AbstractMethod:
        """ get module or the first available clone, using preference and avoiding None """
        if prefer_clone:
            for clone in self.get_method_clones():
                return clone.get_method()
        return self.get_method()

    def iterate_methods_on_device(self) -> Iterable[Tuple[AbstractMethod, str]]:
        """
        iterate the methods that are placed on the main device
        :return: pairs of (method, format string for log_dicts)
        """
        yield self.get_method(), '%s'
        for clone in self.get_method_clones():
            if clone.is_on_same_device():
                yield clone.get_method(), '%s/clones/%s' % ('%s', clone.get_name())

    def get_checkpoint_update_dict(self, *_) -> dict:
        """ get the internal state """
        # optional argument required for lightning
        return {'trainer_state': self._get_state_dict()}

    def _get_state_dict(self) -> dict:
        """ get the internal state """
        return dict()

    def _load_state_dict(self, state: dict):
        """ load the internal state """
        pass
