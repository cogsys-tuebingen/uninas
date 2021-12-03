"""
common estimator (metric) utils to rank different networks (architecture subsets of a super network)
"""

import torch
from uninas.methods.abstract_method import AbstractMethod
from uninas.models.networks.uninas.search import SearchUninasNetwork
from uninas.training.trainer.simple import SimpleTrainer
from uninas.optimization.estimators.abstract import AbstractEstimator
from uninas.utils.args import ArgsInterface, Argument, Namespace
from uninas.utils.torch.misc import count_parameters
from uninas.register import Register


class AbstractNetEstimator(AbstractEstimator, ArgsInterface):
    def __init__(self, args: Namespace, index=None,
                 trainer: SimpleTrainer = None, load_path: str = None,
                 method: AbstractMethod = None, **kwargs):
        super().__init__(args, index=index, **kwargs)

        # ensure all required parameters are available
        if (method is None) and (trainer is not None):
            method = trainer.get_method()
        reg_kwargs = Register.get_my_kwargs(self.__class__)
        if reg_kwargs.get('requires_trainer'):
            assert isinstance(trainer, SimpleTrainer), "%s needs a trainer" % self.__class__.__name__
            assert isinstance(load_path, str), "%s needs a path to load weights" % self.__class__.__name__
        if reg_kwargs.get('requires_method'):
            assert isinstance(method, AbstractMethod), "%s needs a method" % self.__class__.__name__
            assert isinstance(method.get_network(), SearchUninasNetwork),\
                "%s's method must use a search network" % self.__class__.__name__

        self.method = method
        self.trainer = trainer
        self.load_path = load_path

    @property
    def net(self) -> SearchUninasNetwork:
        net = self.trainer.get_network()
        assert isinstance(net, SearchUninasNetwork)
        return net

    def short_name(self) -> str:
        raise NotImplementedError

    def _evaluate_tuple(self, values: tuple, strategy_name: str = None) -> float:
        """
        NOTE: either this or the _evaluate_batch method must be implemented in subclasses
        evaluate a single tuple

        :param values: tuple
        :param strategy_name: None if candidate is global, otherwise specific to this weight strategy
        :return: single float value of how well the given parameter values do
        """
        self.net.set_forward_strategy(False)
        if isinstance(strategy_name, str):
            self.net.forward_strategy(strategy_dict={strategy_name: dict(fixed_arc=values)})
        else:
            self.net.forward_strategy(fixed_arc=values)
        return self._evaluate_tuple_in_net(values=values)

    def _evaluate_tuple_in_net(self, values: tuple):
        raise NotImplementedError


@Register.hpo_estimator(requires_method=True)
class NetParamsEstimator(AbstractNetEstimator):
    """
    Estimating the network parameter count
    (does not properly work with paths that share weights)
    (does not properly work with shared architecture weights)
    (does not account for partial network evaluation yet, as needed in blockwisely search/evaluation)

    Can also be achieved by profiling params, then using a fitted ModelEstimator (has the same drawbacks)
    """

    def __init__(self, *args_, **kwargs_):
        super().__init__(*args_, **kwargs_)
        self.is_set_up = False
        self.choices = []
        self.const = 0

    def short_name(self) -> str:
        return "parameters"

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('count_only_trainable', default='True', type=str, help='ignore buffers etc', is_bool=True),
        ]

    def get_params(self, cell_index: int, choice_index: int) -> int:
        try:
            return self.choices[cell_index][choice_index]
        except:
            return 0

    def _evaluate_tuple_in_net(self, values: tuple):
        if not self.is_set_up:
            self.is_set_up = True
            checkpoint = torch.load(self.load_path)
            state_dict = checkpoint.get('state_dict', None)
            added_state = checkpoint.get('net_add_state', None)
            count_only_trainable = self.kwargs['count_only_trainable']
            assert state_dict is not None, added_state is not None
            # stem / head weights are in every model
            if count_only_trainable:
                self.const += count_parameters(self.method.get_network().get_network().get_stem())
                self.const += count_parameters(self.method.get_network().get_network().get_heads())
            else:
                for k, v in state_dict.items():
                    if '.stem.' in k or '.heads.' in k:
                        self.const += torch.numel(v)
            # variable num params depending on gene
            for choices in added_state.get('cells', list()):
                num_params = [0]*len(choices)
                for j, choice in enumerate(choices):
                    for name, shape, trainable in choice:
                        if trainable or not count_only_trainable:
                            num_params[j] += torch.numel(state_dict[name])
                self.choices.append(num_params)
        num_params = sum(self.get_params(i, g) for i, g in enumerate(values)) + self.const
        return num_params


@Register.hpo_estimator(requires_method=True)
class NetMacsEstimator(AbstractNetEstimator):
    """
    Estimating the network MACs
    (does not account for partial network evaluation yet, as needed in blockwisely search/evaluation)

    Can also be achieved by profiling macs, then using a fitted ModelEstimator
    """

    def short_name(self) -> str:
        return "macs"

    def _evaluate_tuple_in_net(self, values: tuple):
        return self.method.profile_macs()


@Register.hpo_estimator(requires_trainer=True, requires_method=True)
class NetValueEstimator(AbstractNetEstimator):
    """
    An Estimator for a value returned by forward passes (loss, accuracy, ...)
    """

    def __init__(self, *args_, **kwargs_):
        # can set self.net_kwargs to account for partial evaluation (only specific blocks)
        super().__init__(*args_, **kwargs_)
        self.net_kwargs = {}

    def short_name(self) -> str:
        return "value(%s)" % self.kwargs['value']

    def set_net_kwargs(self, **kwargs):
        self.net_kwargs = kwargs

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('load', default="False", type=str, help='load the cached weights or continue', is_bool=True),
            Argument('batches_forward', default=0, type=int, help='num batches to forward the network, to adapt bn'),
            Argument('batches_train', default=0, type=int, help='num batches to train the network, -1 for an epoch'),
            Argument('batches_eval', default=-1, type=int, help='num batches to train the network, -1 for an epoch'),
            Argument('value', default='val/accuracy/1', type=str, help='which top k value to optimize'),
        ]

    def _evaluate_tuple_in_net(self, values: tuple):
        if self.kwargs.get('load', False):
            self.trainer.load(self.load_path)
        train, eval_, dct = self.kwargs.get('batches_train', 0), self.kwargs.get('batches_eval', 0), {}
        # forward
        if self.kwargs.get('batches_forward') > 0:
            self.trainer.forward_steps(steps=self.kwargs.get('batches_forward'), **self.net_kwargs)
        # train
        if train > 0:
            dct.update(self.trainer.train_steps(steps=train, **self.net_kwargs))
        elif train < 0:
            dct.update(self.trainer.train_epochs(epochs=1, run_eval=False, run_test=False, **self.net_kwargs))
        # eval
        if eval_ > 0:
            dct.update(self.trainer.eval_steps(steps=eval_, **self.net_kwargs))
        elif eval_ < 0:
            dct.update(self.trainer.eval_epoch(**self.net_kwargs))
            # print(values, self.trainer.get_method().strategy_manager.get_all_finalized_indices(unique=True))
        # return
        v = dct.get(self.kwargs['value'], None)
        if v is None:
            raise KeyError("key %s not in dict, have keys=(%s)" % (self.kwargs['value'], list(dct.keys())))
        return v.cpu().detach().item()
