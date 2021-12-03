import torch
from collections import defaultdict
from uninas.data.abstract import AbstractDataSet
from uninas.models.networks.abstract import AbstractNetwork
from uninas.training.result import ResultValue
from uninas.utils.args import ArgsInterface, Namespace, Argument


class AbstractMetric(ArgsInterface):
    """
    Metrics during (supervised) network training,
    between network outputs and some targets
    """

    def __init__(self, head_weights: list, **kwargs):
        super().__init__()
        self.head_weights = head_weights
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def get_log_name(self) -> str:
        return self.__class__.__name__

    @classmethod
    def from_args(cls, args: Namespace, index: int, data_set: AbstractDataSet, head_weights: list) -> 'AbstractMetric':
        """
        :param args: global arguments namespace
        :param index: index of this metric
        :param data_set: data set that is evaluated on
        :param head_weights: how each head is weighted
        """
        all_parsed = cls._all_parsed_arguments(args, index=index)
        return cls(head_weights=head_weights, **all_parsed)

    @classmethod
    def _to_dict(cls, key: str, prefix: str, name: str, dct: dict) -> dict:
        """ adds key and name to all dict entries """
        s = "%s/%s" % (key, name) if len(prefix) == 0 else "%s/%s/%s" % (prefix, key, name)
        return {'%s/%s' % (s, k): v for k, v in dct.items()}

    @classmethod
    def _batchify_tensors(cls, logits: [torch.Tensor], targets: torch.Tensor) -> ([torch.Tensor], torch.Tensor):
        """
        reshape all [batch, classes, n0, n1, ...] tensors into [batch, classes]
        :param logits: network outputs
        :param targets: output targets
        """
        new_logits = []
        for tensor in logits + [targets]:
            shape = tensor.shape
            if len(shape) > 2:
                new_logits.append(tensor.transpose(0, 1).reshape(shape[1], -1).transpose(0, 1))
            else:
                new_logits.append(tensor)
        return new_logits[:-1], new_logits[-1]

    @classmethod
    def _remove_onehot(cls, targets: torch.Tensor) -> torch.Tensor:
        """ remove one-hot encoding from a [batch, classes] tensor """
        if len(targets.shape) == 2:
            return torch.argmax(targets, dim=-1)
        return targets

    @classmethod
    def _ignore_with_index(cls, logits: [torch.Tensor], targets: torch.Tensor,
                           ignore_target_index=-999, ignore_prediction_index=-999) ->\
            ([torch.Tensor], torch.Tensor):
        """
        remove all occurrences where the target equals the ignore index, prevent logits from predicting an ignored class

        :param logits: network outputs, each has the [batch, classes] shape
        :param targets: output targets, has the [batch] shape
        :param ignore_target_index: remove all samples where the target matches this index
        :param ignore_prediction_index: if the network predicts this index, choose the next most-likely prediction instead
        """
        # remove all occurrences where the target equals the ignore index
        if ignore_target_index >= 0:
            to_use = targets != ignore_target_index
            logits = [lg[to_use] for lg in logits]
            targets = targets[to_use]

        # prevent logits from predicting an ignored class
        if ignore_prediction_index >= 0:
            new_logits = [lg.clone().detach_() for lg in logits]
            for lg in new_logits:
                min_ = lg.min(axis=1).values
                lg[:, ignore_prediction_index] = min_
            logits = new_logits

        return logits, targets

    def get_accumulated_stats(self, key: str) -> {str: torch.Tensor}:
        """ get the averaged statistics for a specific key """
        return {}

    def eval_accumulated_stats(self, save_dir: str, key: str, prefix="", epoch: int = None, stats: dict = None) -> dict:
        """
        visualize/log this metric

        :param save_dir: if stats are visualized, where to save them
        :param key: key to log
        :param prefix: string prefix added in front of each dict key
        :param epoch: optional int
        :param stats: {str: tensor} or {str: [tensor]}
        :return: usually empty dict if stats are visualized, otherwise the result of accumulating the stats
        """
        return {}

    def reset(self, key: str = None):
        """ reset tracked stats for a specific key, or all (if key == None) """
        pass

    def on_epoch_start(self, epoch: int, is_last=False):
        pass

    def evaluate(self, net: AbstractNetwork,
                 inputs: torch.Tensor, logits: [torch.Tensor], targets: torch.Tensor, key: str) -> {str: ResultValue}:
        """

        :param net: evaluated network
        :param inputs: network inputs
        :param logits: network outputs
        :param targets: output targets
        :param key: prefix for the dict keys, e.g. "train" or "test"
        :return: dictionary of string keys with corresponding results
        """
        raise NotImplementedError

    def _evaluate(self, net: AbstractNetwork, inputs: torch.Tensor,
                  logits: [torch.Tensor], targets: torch.Tensor) -> {str: ResultValue}:
        """

        :param net: evaluated network
        :param inputs: network inputs
        :param logits: network outputs
        :param targets: output targets
        :return: dictionary of string keys with corresponding results
        """
        raise NotImplementedError


class AbstractLogMetric(AbstractMetric):
    """
    A metric that is logged epoch-wise to the output stream and loggers (e.g. tensorboard),
    all single results of _evaluate() are weighted averaged later, by how the batch sizes of each single result
    """

    def evaluate(self, net: AbstractNetwork,
                 inputs: torch.Tensor, logits: [torch.Tensor], targets: torch.Tensor, key: str) -> {str: ResultValue}:
        """

        :param net: evaluated network
        :param inputs: network inputs
        :param logits: network outputs
        :param targets: output targets
        :param key: prefix for the dict keys, e.g. "train" or "test"
        :return: dictionary of string keys with corresponding results
        """
        with torch.no_grad():
            cur = self._evaluate(net, inputs, logits, targets)
            cur = {k: v.unsqueeze() for k, v in cur.items()}
            return self._to_dict(key, "", self.get_log_name(), cur)

    def _evaluate(self, net: AbstractNetwork, inputs: torch.Tensor,
                  logits: [torch.Tensor], targets: torch.Tensor) -> {str: ResultValue}:
        """

        :param net: evaluated network
        :param inputs: network inputs
        :param logits: network outputs
        :param targets: output targets
        :return: dictionary of string keys with corresponding results
        """
        raise NotImplementedError


class AbstractAccumulateMetric(AbstractMetric):
    """
    A metric that accumulates stats first
    """

    def __init__(self, head_weights: list, each_epochs=-1, **kwargs):
        super().__init__(head_weights, **kwargs)
        self.stats = defaultdict(dict)
        self.each_epochs = each_epochs
        self.is_active = False

    @classmethod
    def _combine_tensors(cls, dict_key: str, tensors: [torch.Tensor]) -> torch.Tensor:
        """ how to combine tensors if they are gathered from distributed training or from different batches """
        return sum(tensors)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('each_epochs', default=-1, type=int, help='visualize each n epochs, only last if <=0'),
        ]

    def reset(self, key: str = None):
        """ reset tracked stats for a specific key, or all (if key == None) """
        keys = [key] if isinstance(key, str) else list(self.stats.keys())
        for k in keys:
            self.stats[k].clear()

    def on_epoch_start(self, epoch: int, is_last=False):
        self.reset(key=None)
        self.is_active = is_last or ((self.each_epochs > 0) and ((epoch + 1) % self.each_epochs == 0))

    def evaluate(self, net: AbstractNetwork,
                 inputs: torch.Tensor, logits: [torch.Tensor], targets: torch.Tensor, key: str) -> {str: torch.Tensor}:
        """

        :param net: evaluated network
        :param inputs: network inputs
        :param logits: network outputs
        :param targets: output targets
        :param key: prefix for the dict keys, e.g. "train" or "test"
        :return: dictionary of string keys with corresponding [scalar] tensors
        """
        if not self.is_active:
            return {}

        with torch.no_grad():
            cur = self._evaluate(net, inputs, logits, targets)

            # add all values to current stat dict
            for k, v in cur.items():
                if k in self.stats[key]:
                    self.stats[key][k] = self._combine_tensors(k, [self.stats[key][k], v.value])
                else:
                    self.stats[key][k] = v.value

            return {}

    def _evaluate(self, net: AbstractNetwork, inputs: torch.Tensor,
                  logits: [torch.Tensor], targets: torch.Tensor) -> {str: ResultValue}:
        """

        :param net: evaluated network
        :param inputs: network inputs
        :param logits: network outputs
        :param targets: output targets
        :return: dictionary of string keys with corresponding results
        """
        raise NotImplementedError

    def get_accumulated_stats(self, key: str) -> {str: torch.Tensor}:
        """ get the averaged statistics for a specific key """
        return self.stats.get(key, {})

    def eval_accumulated_stats(self, save_dir: str, key: str, prefix="", epoch: int = None, stats: dict = None) -> dict:
        """
        visualize/log this metric

        :param save_dir: if stats are visualized, where to save them
        :param key: key to log
        :param prefix: string prefix added in front of each dict key
        :param epoch: optional int
        :param stats: {str: tensor} or {str: [tensor]}
        :return: usually empty dict if stats are visualized, otherwise the result of accumulating the stats
        """
        if stats is None:
            stats = self.get_accumulated_stats(key)
        else:
            with torch.no_grad():
                stats = {k: self._combine_tensors(k, v) if isinstance(v, list) else v for k, v in stats.items()}
        if len(stats) > 0:
            if isinstance(epoch, int):
                save_dir = '%s/epoch_%d/' % (save_dir, epoch)
            save_path = '%s/%s/%s' % (save_dir, prefix, key)
            stats = self._update_stats(stats)
            self._viz_stats(save_path, stats)
            return self._to_dict(key, prefix, self.get_log_name(), self._log_stats(stats))
        return {}

    def _update_stats(self, stats: dict) -> dict:
        """
        pre-compute things on the stats that may be shared across log/viz

        :param stats: accumulated stats throughout the _evaluate calls
        :return: stats
        """
        return stats

    def _log_stats(self, stats: dict) -> dict:
        """
        compute this metric

        :param stats: accumulated stats throughout the _evaluate() calls, after calling _update_stats() on them
        :return: log dict
        """
        return {}

    def _viz_stats(self, save_path: str, stats: dict):
        """
        visualize this metric

        :param save_path: where to save
        :param stats: accumulated stats throughout the _evaluate() calls, after calling _update_stats() on them
        :return:
        """
        pass
