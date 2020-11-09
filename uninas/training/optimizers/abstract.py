"""
abstract optimizer with some default args
"""

import types
from typing import Union, List
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.cuda.amp import GradScaler
from uninas.utils.args import ArgsInterface, Argument


class MultiOptimizer:
    """
    wraps multiple optimizers in one interface to sidestep some lightning restrictions
    """

    # behave like an optimizer

    def __init__(self, optimizers: list):
        self.optimizers = optimizers

    def at_index(self, index: int):
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

    def clip_grad(self, index: int, scaler: GradScaler, value: float, norm_value: float, norm_type: float):
        self.optimizers[index].clip_grad(scaler, value, norm_value, norm_type)

    def step(self, index: int, closure=None):
        self.optimizers[index].step(closure)

    # iterable like a list

    def __iter__(self):
        for o in self.optimizers:
            yield o

    def __len__(self):
        return len(self.optimizers)

    # iterate states as if a dict, done by lightning

    @property
    def state(self):
        return self

    def items(self):
        for o in self.optimizers:
            for k, v in o.state.items():
                yield k, v

    def keys(self):
        for k, v in self.items():
            yield k

    def values(self):
        for k, v in self.items():
            yield v


class AbstractOptimizer(ArgsInterface):
    optimizer_cls = None

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('weight_decay', default=0.0, type=float, help='weight decay'),
            Argument('weight_decay_filter', default='True', type=str, help='filter bias/bn from decay', is_bool=True),
        ]

    @classmethod
    def _decay_and_params(cls, named_params, weight_decay: float, weight_decay_filter: bool) -> (list, float):
        """
        possibly filter bias and bn params so that they don't have weight decay
        the original (?) idea is from https://github.com/rwightman/pytorch-image-models
        """
        if weight_decay_filter and weight_decay > 0.0:
            decay, no_decay = [], []
            for n, param in named_params:
                # keep params with no_grad, may be enabled later
                if len(param.shape) == 1 or n.endswith('.bias'):
                    no_decay.append(param)
                else:
                    decay.append(param)
            if len(decay) == 0:
                return no_decay, 0.0
            if len(no_decay) == 0:
                return decay, weight_decay
            return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)], 0.0
        return [p for _, p in named_params], weight_decay

    @classmethod
    def __new__(cls, *args, **kwargs):
        optimizer, namespace = args
        index, named_params = kwargs.get('index', None), kwargs.get('named_params', None)
        o_args = cls._all_parsed_arguments(args=namespace, index=index)

        # possibly disable weight decay for certain types of parameters
        weight_decay = o_args.pop('weight_decay')
        weight_decay_filter = o_args.pop('weight_decay_filter')
        params, weight_decay = cls._decay_and_params(named_params, weight_decay, weight_decay_filter)
        torch_optimizer = optimizer.optimizer_cls(params=params, **o_args, weight_decay=weight_decay)

        # adding a method to clip gradients directly in the optimizer
        def clip_grad(self, scaler: GradScaler, clip_value=-1, clip_norm_value=-1, clip_norm_type=2):
            if clip_value > 0:
                scaler.unscale_(self)
                for group in self.param_groups:
                    nn.utils.clip_grad_value_(group['params'], clip_value)
            elif clip_norm_value > 0:
                scaler.unscale_(self)
                for group in self.param_groups:
                    nn.utils.clip_grad_norm_(group['params'], clip_norm_value, clip_norm_type)

        torch_optimizer.clip_grad = types.MethodType(clip_grad, torch_optimizer)
        return torch_optimizer

    @classmethod
    def get_optimizer_log_dict(cls, optimizers: [Optimizer]) -> dict:
        assert len(optimizers) == 1, "Only one optimizer is allowed (use %s)" % MultiOptimizer.__class__.__name__
        log_dict = {}
        for j, o in enumerate(optimizers):
            if isinstance(o, MultiOptimizer):
                log_dict.update({'learning_rate/%d/%d/%s' % (j, j2, o2.__class__.__name__): o2.param_groups[0]['lr']
                                 for j2, o2 in enumerate(o.optimizers)})
            else:
                log_dict.update({'learning_rate/%d/%s' % (j, o.__class__.__name__): o.param_groups[0]['lr']})
        return log_dict

    @classmethod
    def filter_values_in_dict(cls, log_dict: dict, optimizer_index: int) -> dict:
        """ return only the log_dict entries that match this optimizer """
        filtered, s = {}, ("learning_rate/%d" % optimizer_index)
        for k, v in log_dict.items():
            if k.startswith(s):
                filtered[k] = v
        return filtered

    @classmethod
    def set_optimizer_lr(cls, optimizer: Optimizer, lr: float, is_multiplier=False):
        """
        set the learning rate and initial learning rate of an optimizer to 'lr'

        :param optimizer:
        :param lr: new (initial) learning rate
        :param is_multiplier: if set just multiply the current value in optimizer with 'lr' instead
        """
        for pg in optimizer.param_groups:
            used_lr = lr * (pg['lr'] if is_multiplier else 1)
            pg['lr'] = used_lr
            pg['initial_lr'] = used_lr

    @classmethod
    def set_optimizer_lr_by_index(cls, optimizers: List[Union[Optimizer, MultiOptimizer]], index: int,
                                  lr: float, is_multiplier=False):
        """
        set the learning rate and initial learning rate of an optimizer to 'lr'

        :param optimizers:
        :param index: index of the optimizer to adapt
        :param lr: new (initial) learning rate
        :param is_multiplier: if set just multiply the current value in optimizer with 'lr' instead
        """
        assert len(optimizers) == 1, "Only one optimizer is allowed (use %s)" % MultiOptimizer.__class__.__name__
        optimizer = optimizers[0]
        if isinstance(optimizer, Optimizer) and index == 0:
            return cls.set_optimizer_lr(optimizer, lr, is_multiplier=is_multiplier)
        elif isinstance(optimizer, MultiOptimizer):
            return cls.set_optimizer_lr(optimizer.at_index(index), lr, is_multiplier=is_multiplier)
        raise NotImplementedError("Only one optimizer available, want to adapt at index %d" % index)
