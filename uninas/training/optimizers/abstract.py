"""
abstract optimizer with some default args
"""

from copy import deepcopy
from typing import Union, List, Tuple, Optional, Callable
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.cuda.amp import GradScaler
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.training.result import LogResult
from uninas.utils.args import ArgsInterface, Argument, Namespace


class AbstractOptimizerClosure:
    """
    Abstract closure (either re-usable or as factory) that can be used by optimizers.
    It has to be prepared with the arguments for the training step (i.e. net input, ...),
    and saves the first result it gets (the results of further calls are discarded).
    """

    def __init__(self, mover: AbstractDeviceMover, training_step: Callable):
        self.mover = mover
        self.training_step = training_step
        self.training_step_args = None
        self.training_step_kwargs = None
        self.num_calls = 0
        self.result = None

    def prepare(self, *training_step_args, **training_step_kwargs) -> 'AbstractOptimizerClosure':
        """
        Add args/kwargs to the training step

        :param training_step_args:
        :param training_step_kwargs:
        """
        self.training_step_args = training_step_args
        self.training_step_kwargs = training_step_kwargs
        self.num_calls = 0
        self.result = None
        return self

    def get_result(self) -> LogResult:
        """
        Get the result after the optimizer is done
        :return: detached LogResult
        """
        assert self.num_calls > 0, "This closure was not called, can not have a result"
        assert isinstance(self.result, LogResult), "The closure does not have a log result!"
        return self.result

    def __call__(self):
        """
        The optimizer calls the closure
        """
        raise NotImplementedError


class AbstractOptimizerFunctions:
    closure_cls = None      # closure class for the optimizer, optional

    def set_optimizer_lr(self, lr: float, update_initial=True, is_multiplier=False, at_index=0):
        """
        set the (initial) learning rate to 'lr'

        :param lr: new (initial) learning rate
        :param update_initial: if 'lr' should also apply to the initial learning rate
        :param is_multiplier: if set just multiply the current value in optimizer with 'lr' instead
        :param at_index: compatibility for multi optimizers
        """
        raise NotImplementedError

    @classmethod
    def set_optimizer_lr_by_index(cls, optimizers: ['AbstractOptimizerFunctions'], index: int, lr: float, is_multiplier=False):
        """
        set the learning rate and initial learning rate of all optimizers to 'lr'

        :param optimizers:
        :param index: index of the optimizer to adapt
        :param lr: new (initial) learning rate
        :param is_multiplier: if set just multiply the current value in optimizer with 'lr' instead
        """
        for optimizer in optimizers:
            optimizer.set_optimizer_lr(lr, update_initial=True, is_multiplier=is_multiplier, at_index=index)

    def get_all_weights(self) -> [nn.Parameter]:
        raise NotImplementedError

    def get_all_gradients(self, make_copy=True) -> List[Union[torch.Tensor, None]]:
        """
        get a list of the gradients optimized
        """
        grads = []
        for p in self.get_all_weights():
            if p.grad is None:
                grads.append(None)
            elif make_copy:
                grads.append(deepcopy(p.grad.detach()))
            else:
                grads.append(p.grad)
        return grads

    def get_my_log_dict(self, my_index=0) -> dict:
        raise NotImplementedError

    @classmethod
    def get_optimizer_log_dict(cls, optimizers: ['AbstractOptimizerFunctions']) -> dict:
        log_dict = {}
        for i, optimizer in enumerate(optimizers):
            log_dict.update(optimizer.get_my_log_dict(my_index=i))
        return log_dict

    @classmethod
    def filter_values_in_dict(cls, log_dict: dict, optimizer_index: int) -> dict:
        """ return only the log_dict entries that match this optimizer """
        filtered, s = {}, ("learning_rate/%d" % optimizer_index)
        for k, v in log_dict.items():
            if k.startswith(s):
                filtered[k] = v
        return filtered

    def get_closure(self, mover: AbstractDeviceMover, training_step: Callable) -> Optional[AbstractOptimizerClosure]:
        """ if this optimizer requires a closure to train, return it, otherwise None """
        if (self.closure_cls is not None) and issubclass(self.closure_cls, AbstractOptimizerClosure):
            return self.closure_cls(mover, training_step)
        return None

    # lightning compatibility

    def items(self):
        raise NotImplementedError

    def keys(self):
        for k, v in self.items():
            yield k

    def values(self):
        for k, v in self.items():
            yield v


class WrappedOptimizer(ArgsInterface, AbstractOptimizerFunctions, Optimizer):
    optimizer_cls = None    # callable to create a torch optimizer

    # creation

    def __init__(self, torch_optimizer: Optimizer, scaler: GradScaler,
                 clip_abs_value: float = -1, clip_norm_value: float = -1, clip_norm_type: float = -1):
        super().__init__()
        self.torch_optimizer = torch_optimizer
        self.scaler = scaler

        self.clip_abs_value = clip_abs_value
        self.clip_norm_value = clip_norm_value
        self.clip_norm_type = clip_norm_type
        self._do_clip_abs = self.clip_abs_value > 0
        self._do_clip_norm = (self.clip_norm_value > 0) and (self.clip_norm_type > 0)
        assert not (self._do_clip_abs and self._do_clip_norm), "Can not clip both, absolute and norm values"

    @classmethod
    def from_args(cls, namespace: Namespace, index: Union[int, None] = 0, scaler: Union[GradScaler, None] = None,
                  named_params: Union[List[tuple], Tuple[tuple]] = (()), kwargs_changes: dict = None):
        o_args = cls._all_parsed_arguments(args=namespace, index=index)
        if isinstance(kwargs_changes, dict):
            o_args.update(kwargs_changes)

        # grad scaler
        if not isinstance(scaler, GradScaler):
            scaler = GradScaler(enabled=False)

        # clipping
        clip_abs_value = o_args.pop('clip_abs_value')
        clip_norm_value = o_args.pop('clip_norm_value')
        clip_norm_type = o_args.pop('clip_norm_type')

        # possibly disable weight decay for certain types of parameters
        weight_decay = o_args.pop('weight_decay')
        weight_decay_filter = o_args.pop('weight_decay_filter')
        params, weight_decay = cls._decay_and_params(named_params, weight_decay, weight_decay_filter)
        torch_optimizer = cls.optimizer_cls(params=params, **o_args, weight_decay=weight_decay)

        return cls(torch_optimizer, scaler,
                   clip_abs_value=clip_abs_value, clip_norm_value=clip_norm_value, clip_norm_type=clip_norm_type)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('weight_decay', default=0.0, type=float, help='weight decay'),
            Argument('weight_decay_filter', default='True', type=str, help='filter bias/bn and architecture weights from decay', is_bool=True),
            Argument('clip_abs_value', default=-1, type=float, help='clip gradient to +- value, <=0 to disable'),
            Argument('clip_norm_value', default=-1, type=float, help='clip gradient norm value, <=0 to disable'),
            Argument('clip_norm_type', default=2, type=float, help='clip gradient norm type'),
        ]

    @classmethod
    def _decay_and_params(cls, named_params: list, weight_decay: float, weight_decay_filter: bool) -> (list, float):
        """
        possibly filter bias, bn and architecture params so that they don't have weight decay
        the original (?) idea is from https://github.com/rwightman/pytorch-image-models
        """
        if weight_decay_filter and weight_decay > 0.0:
            decay, no_decay = [], []
            for n, param in named_params:
                # keep params with no_grad, may be enabled later
                if len(param.shape) == 1 or n.endswith('.bias') or n.startswith('strategies'):
                    no_decay.append(param)
                else:
                    decay.append(param)
            if len(decay) == 0:
                return no_decay, 0.0
            if len(no_decay) == 0:
                return decay, weight_decay
            return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)], 0.0
        return [p for _, p in named_params], weight_decay

    # imitate an optimizer, execute the wrapped torch optimizer

    @property
    def param_groups(self):
        return self.torch_optimizer.param_groups

    def __getstate__(self):
        return self.torch_optimizer.__getstate__()

    def __setstate__(self, state):
        self.torch_optimizer.__setstate__(state)

    def __repr__(self) -> str:
        clip_str, acc_str = '', ''
        if self._do_clip_norm:
            clip_str = 'clip_norm=(%.2f, %.2f), ' % (self.clip_norm_value, self.clip_norm_type)
        if self._do_clip_abs:
            clip_str = 'clip_abs=%.2f, ' % self.clip_abs_value
        return '%s ( %s%s%s )' % (self.__class__.__name__, clip_str, acc_str, self.torch_optimizer.__repr__())

    def state_dict(self):
        return self.torch_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.torch_optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none=False):
        self.torch_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        # unscale
        self.scaler.unscale_(self.torch_optimizer)
        # clip gradients, then step
        if self._do_clip_abs:
            for group in self.param_groups:
                nn.utils.clip_grad_value_(group['params'], self.clip_abs_value)
        if self._do_clip_norm:
            for group in self.param_groups:
                nn.utils.clip_grad_norm_(group['params'], self.clip_norm_value, self.clip_norm_type)
        # step
        self.scaler.step(optimizer=self.torch_optimizer, closure=closure)

    def add_param_group(self, param_group):
        self.torch_optimizer.add_param_group(param_group)

    # utilities

    def get_lr_ratio(self) -> float:
        """ get ratio of current to initial learning rate """
        return self.param_groups[0]['lr'] / self.param_groups[0]['initial_lr']

    def set_optimizer_lr(self, lr: float, update_initial=True, is_multiplier=False, at_index=0):
        """
        set the (initial) learning rate to 'lr'

        :param lr: new (initial) learning rate
        :param update_initial: if 'lr' should also apply to the initial learning rate
        :param is_multiplier: if set just multiply the current value in optimizer with 'lr' instead
        :param at_index: compatibility for multi optimizers
        """
        assert at_index == 0
        for pg in self.torch_optimizer.param_groups:
            used_lr = lr * (pg['lr'] if is_multiplier else 1)
            pg['lr'] = used_lr
            if update_initial:
                pg['initial_lr'] = used_lr

    def get_all_weights(self) -> [nn.Parameter]:
        """
        get a list of the parameters optimized by the given optimizer
        """
        params = []
        for group in self.torch_optimizer.param_groups:
            params.extend(group['params'])
        return params

    def get_my_log_dict(self, my_index=0) -> dict:
        return {'learning_rate/%d/%s' % (my_index, self.__class__.__name__): self.param_groups[0]['lr']}

    # lightning compatibility

    @property
    def state(self) -> dict:
        return self.torch_optimizer.state

    def items(self):
        return self.torch_optimizer.state.items()


class MultiWrappedOptimizer(AbstractOptimizerFunctions):
    """
    wraps multiple optimizers in one interface to sidestep some lightning restrictions
    """

    # behave like an optimizer

    def __init__(self, optimizers: [WrappedOptimizer]):
        self.optimizers = optimizers
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

    def at_index(self, index: int) -> WrappedOptimizer:
        return self.optimizers[index]

    def __getstate__(self):
        return [o.__getstate__() for o in self.optimizers]

    def __setstate__(self, state):
        for s, o in zip(state, self.optimizers):
            o.__setstate__(s)

    def state_dict(self):
        dicts = [o.state_dict() for o in self.optimizers]
        return {
            'state': [d['state'] for d in dicts],
            'param_groups': [d['param_groups'] for d in dicts],
        }

    def load_state_dict(self, state_dict):
        for o, s, p in zip(self.optimizers, state_dict['state'], state_dict['param_groups']):
            o.load_state_dict(dict(state=s, param_groups=p))

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ',\n'.join([repr(o) for o in self.optimizers]))

    def add_param_group(self, index, param_group):
        self.optimizers[index].add_param_group(param_group)

    def zero_grad_all(self):
        for o in self.optimizers:
            o.zero_grad()

    def zero_grad(self, index: int):
        self.optimizers[index].zero_grad()

    def step(self, index: int, closure=None) -> bool:
        return self.optimizers[index].step(closure)

    def step_all(self, closure=None):
        for o in self.optimizers:
            o.step(closure=closure)

    def get_closure(self, mover: AbstractDeviceMover, training_step: Callable) -> Optional[AbstractOptimizerClosure]:
        """ if this optimizer requires a closure to train, return it, otherwise None """
        closures = [opt.get_closure(mover, training_step) for opt in self.optimizers]
        assert all([c is None for c in closures]), "Closures are not implemented for MultiWrappedOptimizer yet"
        return None

    # utilities

    def set_optimizer_lr(self, lr: float, update_initial=True, is_multiplier=False, at_index=0):
        """
        set the (initial) learning rate to 'lr'

        :param lr: new (initial) learning rate
        :param update_initial: if 'lr' should also apply to the initial learning rate
        :param is_multiplier: if set just multiply the current value in optimizer with 'lr' instead
        :param at_index: compatibility for multi optimizers
        """
        self.at_index(at_index).set_optimizer_lr(lr, update_initial=update_initial, is_multiplier=is_multiplier)

    def get_all_weights(self) -> [nn.Parameter]:
        weights = []
        for optimizer in self.optimizers:
            weights.extend(optimizer.get_all_weights)
        return weights

    def get_my_log_dict(self, my_index=0) -> dict:
        return {'learning_rate/%d/%d/%s' % (my_index, i, o.__class__.__name__): o.param_groups[0]['lr']
                for i, o in enumerate(self.optimizers)}

    # iterable like a list

    def __iter__(self):
        for o in self.optimizers:
            yield o

    def __len__(self):
        return len(self.optimizers)

    # iterate states as if a dict, done by lightning

    @property
    def state(self) -> 'MultiWrappedOptimizer':
        return self

    def items(self):
        for o in self.optimizers:
            for k, v in o.state.items():
                yield k, v
