import numpy as np
import torch
from uninas.utils.args import Argument
from uninas.methods.abstract_method import AbstractOptimizationMethod
from uninas.methods.strategy_manager import StrategyManager
from uninas.methods.random import RandomChoiceStrategy
from uninas.register import Register


@Register.strategy(single_path=True)
class MdlStrategy(RandomChoiceStrategy):
    """
    Multinomial Distribution Learning for Effective Neural Architecture Search
    https://arxiv.org/abs/1905.07529
    """

    def __init__(self, max_epochs: int, name='default', key='val/accuracy/1', alpha=0.01, grace_epochs=0):
        super().__init__(max_epochs, name)
        self.key = key
        self.key_start = key.split('/')[0]
        self.alpha = alpha
        self.grace_epochs = grace_epochs
        self._cached_probabilities = {}
        self._cached_sum_values = {}    # per weight: np.array of accumulated op value sum
        self._cached_epoch_count = {}   # per weight: np.array of num op selected
        self._cached_step_count = {}    # per weight: np.array of trained steps

    def build(self):
        for r in self._ordered_unique:
            nc = r.num_choices()
            self._cached_idx[r.name] = 0
            self._cached_max_idx[r.name] = nc
            self._cached_idx_choices[r.name] = np.arange(0, r.num_choices())
            self._cached_probabilities[r.name] = np.ones(shape=(nc,)) / nc
            self._cached_sum_values[r.name] = np.zeros(shape=(nc,), dtype=np.float32)
            self._cached_epoch_count[r.name] = np.zeros(shape=(nc,), dtype=np.int32)
            self._cached_step_count[r.name] = np.zeros(shape=(nc,), dtype=np.int32)

    def get_log_dict(self) -> {str: float}:
        """
        :return: dict of values that are interesting to log
        """
        dct = super().get_log_dict()
        dct.update({"max_value/%s" % n: max(self._cached_probabilities[n]) for n in self.get_weight_names()})
        return dct

    def get_weight_sm(self, name: str) -> torch.Tensor:
        """ softmax over the specified weight """
        return torch.from_numpy(self._cached_probabilities[name])

    def randomize_weights(self):
        """ randomizes all arc weights """
        for n in self.get_weight_names():
            self._cached_idx[n] = np.random.choice(self._cached_max_idx[n], p=self._cached_probabilities[n])

    def get_finalized_indices(self, name: str) -> [int]:
        """ return indices of the modules that should constitute the new architecture, for this specific weight """
        return [np.argmax(self._cached_probabilities[name])]

    def on_epoch_end(self, current_epoch: int) -> bool:
        """
        whenever the method ends an epoch
        signal early stopping when returning True
        """
        # before valid/test feedback
        if current_epoch >= self.grace_epochs:
            for n in self.get_weight_names():
                avg_sum = self._cached_sum_values[n]
                avg_count = self._cached_epoch_count[n]
                avg = np.divide(avg_sum, avg_count, out=np.zeros_like(avg_sum), where=avg_count != 0)
                steps = self._cached_step_count[n]

                # NxN matrices, so that delta_he[i,j] = count[i] - count[j]
                delta_he = np.outer(steps, np.ones_like(steps)) - np.outer(np.ones_like(steps), steps)
                delta_ha = np.outer(avg, np.ones_like(avg)) - np.outer(np.ones_like(avg), avg)

                # conditions for better / worse, counter for better - worse
                cond_b = np.logical_and(delta_he < 0, delta_ha > 0)
                cond_w = np.logical_and(delta_he > 0, delta_ha < 0)
                delta = np.sum(cond_b, axis=1) - np.sum(cond_w, axis=1)

                # update, ensure consistency
                self._cached_probabilities[n] += self.alpha * delta
                if np.min(self._cached_probabilities[n]) < 0:
                    self._cached_probabilities[n] -= np.min(self._cached_probabilities[n])
                    self._cached_probabilities[n] /= np.sum(self._cached_probabilities[n])

                # reset stats
                avg_sum.fill(0.0)
                avg_count.fill(0)
        return False

    def _mask_index(self, idx: int, weight_name: str):
        self._cached_probabilities[weight_name][idx] = 0
        self._cached_probabilities[weight_name] /= np.sum(self._cached_probabilities[weight_name])

    def feedback(self, key: str, log_dict: dict, current_epoch: int, batch_idx: int):
        """
        feedback after each training forward pass
        this is currently not synchronized in distributed training

        :param key: train/val/test
        :param log_dict: contains loss, metric results, ...
        :param current_epoch:
        :param batch_idx:
        :return:
        """
        if key.startswith('train'):
            for n in self.get_weight_names():
                idx = self._cached_idx[n]
                self._cached_step_count[n][idx] += 1
        if key.startswith(self.key_start) and current_epoch >= self.grace_epochs:
            value = log_dict.get(self.key, None)
            assert value is not None, 'The dict does not contain the required key "%s"' % self.key
            for n in self.get_weight_names():
                idx = self._cached_idx[n]
                self._cached_sum_values[n][idx] += value.item()
                self._cached_epoch_count[n][idx] += 1


@Register.method(search=True, single_path=True)
class MdlSearchMethod(AbstractOptimizationMethod):
    """
    Randomly sample 1 out of the available options,
    use validation feedback to rank them

    Multinomial Distribution Learning for Effective Neural Architecture Search
    https://arxiv.org/abs/1905.07529
    """

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('key', default='val/accuracy/1', type=str, help='key to optimize'),
            Argument('alpha', default=0.01, type=float, help='update rate for probability distributions'),
            Argument('grace_epochs', default=0, type=int, help='grace epochs before probability updates'),
        ]

    def setup_strategy(self) -> StrategyManager:
        """ set up the strategy for architecture weights """
        key, alpha, grace_epochs = self._parsed_arguments(['key', 'alpha', 'grace_epochs'], self.hparams)
        return StrategyManager().add_strategy(
            MdlStrategy(self.max_epochs, key=key, alpha=alpha, grace_epochs=grace_epochs))
