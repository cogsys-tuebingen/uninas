"""
regularizing networks during training
"""

from uninas.training.regularizers.abstract import AbstractRegularizer
from uninas.models.networks.abstract import AbstractNetwork
from uninas.modules.modules.misc import DropPathModule
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


@Register.regularizer()
class DropOutRegularizer(AbstractRegularizer):
    """
    Dropout for each head, note that the head itself can also specify a default dropout chance when it is built/saved,
    even if no dropout is desired, it is thus safest to add this regularizer and use probability 0
    """

    def __init__(self, args: Namespace, index=None):
        super().__init__(args, index)
        self.prob = self._parsed_argument('prob', args, self.index)

    def on_epoch_start(self, cur_epoch: int, max_epochs: int, net: AbstractNetwork) -> dict:
        dct = {self._dict_key('p'): self.prob}
        if self._changed:
            self._changed = False
            dct['num'] = n = net.set_dropout_rate(p=self.prob)
            assert n > 0, "Should set dropout probabilities, but found no dropout layer to affect"
        return dct

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('prob', default=0.0, type=float, help='constant dropout probability for heads'),
        ]

    def _str_dict(self) -> dict:
        dct = super()._str_dict()
        dct.update({
            'prob': self.prob,
        })
        return dct

    def _set_value(self, v):
        """ externally set the main value of this regularizer, here the dropout probability """
        assert isinstance(v, float)
        self.prob = v


@Register.regularizer()
class DropPathRegularizer(AbstractRegularizer):
    """
    Linearly increase the chance to drop paths from 'min_prob' to 'max_prob' over the entire training.
    """

    def __init__(self, args: Namespace, index=None):
        # This regularizer is made to work in combination with DropPathModule, which provides the necessary functions
        super().__init__(args, index)
        self.min_prob, self.max_prob = self._parsed_arguments(['min_prob', 'max_prob'], args, self.index)
        self.drop_id_paths = self._parsed_argument('drop_id_paths', args, self.index)
        assert self.max_prob > self.min_prob >= 0,\
            "Strange %s probabilities: max=%f, min=%f" % (self.__class__.__name__, self.max_prob, self.min_prob)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('min_prob', default=0.0, type=float, help='drop path prob, lin increase from this value'),
            Argument('max_prob', default=0.0, type=float, help='drop path prob, lin increase to this value'),
            Argument('drop_id_paths', default='False', type=str, help='can drop identity paths', is_bool=True),
        ]

    def on_start(self, max_epochs: int, net: AbstractNetwork) -> dict:
        num = 0
        for m in net.base_modules_by_condition(lambda m2: isinstance(m2, DropPathModule), recursive=True):
            m.set_drop_ids(self.drop_id_paths)
            num += 1
        assert num > 0, "The network contains no %s, can not set path dropout rate" % DropPathModule.__name__
        return {self._dict_key('drop_id_paths'): 1 if self.drop_id_paths else 0,
                self._dict_key('num_modules'): num}

    def on_epoch_start(self, cur_epoch: int, max_epochs: int, net: AbstractNetwork) -> dict:
        dp = (self.max_prob - self.min_prob) * cur_epoch / max_epochs + self.min_prob
        for m in net.base_modules_by_condition(lambda m2: isinstance(m2, DropPathModule), recursive=True):
            m.set_drop_rate(dp)
        return {self._dict_key('p'): dp}

    def _str_dict(self) -> dict:
        dct = super()._str_dict()
        dct.update({
            'min prob': self.min_prob,
            'max prob': self.max_prob,
            'drop ids': self.drop_id_paths,
        })
        return dct

    def _set_value(self, v):
        """ externally set the main value of this regularizer """
        raise NotImplementedError
