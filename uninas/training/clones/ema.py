import torch
from uninas.methods.abstract_method import AbstractMethod
from uninas.training.clones.abstract import AbstractMethodClone
from uninas.utils.args import Argument
from uninas.register import Register


@Register.training_clone()
class EMAClone(AbstractMethodClone):
    """
    Updates an Exponential Moving Average weight copy of the model, using given 'decay' on the given 'device'
    Inspired by https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py
    """

    def __init__(self, decay=0.999, **kwargs):
        assert 1 > decay > 0
        super().__init__(**kwargs)
        self.decay = decay
        self.decay_m = 1 - decay

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('decay', default=0.999, type=float, help='EMA decay, how much to weight the old info'),
        ]

    def get_name(self) -> str:
        """ name used for log dicts """
        return '%s/%s' % (self.__class__.__name__, self.decay)

    @torch.no_grad()
    def on_update(self, original: AbstractMethod):
        """ whenever the weights of the original method are updated """
        cur_state = self.get_method().get_network().state_dict()
        original_state = original.get_network().state_dict()
        self._update_rec(cur_state, original_state)

    def _update_rec(self, cur_state: dict, original_state: dict):
        for cs, os in zip(cur_state.values(), original_state.values()):
            if isinstance(cs, dict):
                self._update_rec(cs, os)
            else:
                cs.copy_(os.to(self._device) * self.decay_m + cs * self.decay)
