"""
common interface to internal and external networks
"""


from typing import Union
import torchprofile
import numpy as np
import torch
import torch.nn as nn
from uninas.models.abstract import AbstractModel
from uninas.modules.modules.abstract import AbstractArgsModule
from uninas.utils.args import Namespace, Argument
from uninas.utils.shape import Shape, ShapeList
from uninas.utils.torch.misc import count_parameters
from uninas.utils.torch.decorators import use_eval
from uninas.register import Register


class AbstractNetwork(AbstractModel, AbstractArgsModule):
    def __init__(self, model_name: str, checkpoint_path: str, assert_output_match: bool,
                 shape_in_list: [int] = None, shape_out_list: [int] = None):
        super().__init__()
        self._add_to_kwargs(model_name=model_name, checkpoint_path=checkpoint_path,
                            assert_output_match=assert_output_match,
                            shape_in_list=shape_in_list, shape_out_list=shape_out_list)
        self._add_to_print_kwargs(shape_in=None, shape_out=None)
        self._loaded_weights = False
        self._forward_fun = 0
        self._forward_mode_names = {
            'default': 0,
            'cells': 1,
        }

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> 'AbstractNetwork':
        """
        :param args: global argparse namespace
        :param index: argument index
        """
        raise NotImplementedError

    def _get_model_state(self) -> dict:
        """ get a state dict that can later recover the model """
        return {
            'state_dict': self.state_dict(),
            'add_state_dict': self.save_to_state_dict(),
            'kwargs': self.kwargs(),
            'config': self.config(),
        }

    def _load_state(self, model_state: dict) -> bool:
        """ update the current model with this state """
        assert self.kwargs() == model_state['kwargs'], "Can not load the model, their arguments differ"
        if not self.is_built():
            self.build_from_cache()
        self.load_state_dict(model_state['state_dict'])
        self.load_from_state_dict(model_state['add_state_dict'])
        self.loaded_weights(True)
        return True

    @classmethod
    def _load_from(cls, model_state: dict) -> 'AbstractNetwork':
        """ create this model from a state dict """
        model = cls(**model_state['kwargs'])
        model._load_state(model_state)
        return model

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('checkpoint_path', default='', type=str, is_path=True,
                     help='use pretrained weights within the given local directory (matching by network name) or from an url'),
            Argument('assert_output_match', default='True', type=str, is_bool=True,
                     help='assert that the network output shape (each head) matches the expectations (by the dataset)'),
        ]

    def prepare_predict(self, device: str) -> 'AbstractNetwork':
        """ place the model on some hardware device, go eval mode """
        model = self.to(device) if len(device) > 0 else self
        model.eval()
        return model

    def fit(self, data: np.array, labels: np.array):
        """
        fit the model to data+labels
        :param data: n-dimensional np array, first dimension is the batch
        :param labels: n-dimensional np array, first dimension is the batch
        :return:
        """
        raise TypeError("Network models can not be fit directly")

    def predict(self, data: np.array) -> np.array:
        """
        predict the labels of the data, only from the last head
        :param data: n-dimensional np array, first dimension is the batch
        :return: n-dimensional np array, first dimension is the batch
        """
        with torch.no_grad():
            data = torch.from_numpy(data).to(torch.float32).to(self.get_device())
            return self(data)[-1].cpu().numpy()

    def get_model_name(self) -> str:
        return self.model_name

    def loaded_weights(self, b=True):
        self._loaded_weights = b

    def has_loaded_weights(self) -> bool:
        return self._loaded_weights

    def save_to_state_dict(self) -> dict:
        """ store additional info in the state dict """
        return dict()

    def load_from_state_dict(self, state: dict):
        """ load the stored additional info from the state dict """
        pass

    def get_head_weightings(self) -> [float]:
        """ get the weights of all heads, in order """
        return [1.0]*len(self.get_heads())

    def get_stem_input_shape(self) -> Shape:
        """ input shape of the stem (therefore also the entire net) """
        return self.shape_in

    def get_stem_output_shape(self, flatten=False) -> ShapeList:
        """ output shapes of the stem """
        if self.get_cached('stem_output_shapes') is None:
            self.cached['stem_output_shapes'] = self._get_stem_output_shape()
        shapes = self.get_cached('stem_output_shapes')
        return shapes.flatten(flatten)

    def _get_stem_output_shape(self) -> ShapeList:
        """ output shapes of the stem """
        raise NotImplementedError

    def get_cell_input_shapes(self, flatten=False) -> ShapeList:
        """ input shape(s) of each cell in order """
        if self.get_cached('all_input_shapes') is None:
            self.cached['all_input_shapes'] = self._get_cell_input_shapes()
        shapes = self.get_cached('all_input_shapes')
        return shapes.flatten(flatten)

    def _get_cell_input_shapes(self) -> ShapeList:
        """ input shape(s) of each cell in order """
        raise NotImplementedError

    def get_cell_output_shapes(self, flatten=False) -> ShapeList:
        """ output shape(s) of each cell in order """
        if self.get_cached('all_output_shapes') is None:
            self.cached['all_output_shapes'] = self._get_cell_output_shapes()
        shapes = self.get_cached('all_output_shapes')
        return shapes.flatten(flatten)

    def _get_cell_output_shapes(self) -> ShapeList:
        """ output shape(s) of each cell in order """
        raise NotImplementedError

    def get_heads_input_shapes(self) -> ShapeList:
        """ input shape of the head(s) """
        cell_shapes = self.get_cell_output_shapes()
        if len(cell_shapes) > 0:
            return cell_shapes.shapes[-1]
        return self.get_stem_output_shape()

    def get_network_output_shapes(self, flatten=True) -> ShapeList:
        """ output shapes of the network """
        if self.get_cached('net_output_shapes') is None:
            self.cached['net_output_shapes'] = self._get_network_output_shapes()
        shapes =  self.get_cached('net_output_shapes')
        return shapes.flatten(flatten)

    def _get_network_output_shapes(self) -> ShapeList:
        """ output shapes of the network """
        raise NotImplementedError

    def set_dropout_rate(self, p=None) -> int:
        """ set the dropout rate of every dropout layer to p """
        if isinstance(p, float):
            return self._set_dropout_rate(p)
        return 0

    def _set_dropout_rate(self, p: float) -> int:
        """ set the dropout rate of every dropout layer to p, no change for p=None """
        # set any dropout layer to p
        n = 0
        for m in self.get_network().modules():
            if isinstance(m, nn.Dropout):
                m.p = p
                n += 1
        assert n > 0 or p <= 0, "Could not set the dropout rate to %f, no nn.Dropout modules found!" % p
        return n

    def get_num_parameters(self) -> int:
        return count_parameters(self)

    def get_network(self) -> nn.Module:
        raise NotImplementedError

    def get_stem(self) -> nn.Module:
        raise NotImplementedError

    def num_cells(self) -> int:
        return len(self.get_cells())

    def get_cells(self) -> nn.ModuleList():
        raise NotImplementedError

    def get_heads(self) -> nn.ModuleList():
        raise NotImplementedError

    def build_from_cache(self) -> ShapeList:
        """ build the network from cached input/output shapes """
        shape_in = Shape(self.shape_in_list)
        shape_out = Shape(self.shape_out_list)
        return self.build(shape_in, shape_out)

    def build(self, s_in: Shape, s_out: Shape) -> ShapeList:
        return super().build(s_in, s_out)

    def _build(self, s_in: Shape, s_out: Shape) -> ShapeList:
        raise NotImplementedError

    def use_forward_mode(self, mode='default'):
        """
        :param mode:
            default: default pass from input to all outputs
            cells: execute only specific cells (from i to j)
        """
        v = self._forward_mode_names.get(mode)
        assert v is not None, "unknown mode %s" % mode
        self._forward_fun = v

    def get_forward_mode(self) -> str:
        for k, v in self._forward_mode_names.items():
            if v == self._forward_fun:
                return k
        raise ValueError("Currently using %s which is not implemented..." % str(self._forward_fun))

    def forward(self, *args, **kwargs):
        if self._forward_fun == 0:
            return self.all_forward(*args, **kwargs)
        return self.specific_forward(*args, **kwargs)

    def all_forward(self, x: torch.Tensor) -> [torch.Tensor]:
        """
        returns list of all heads' outputs
        the heads are sorted by ascending cell order
        """
        raise NotImplementedError

    def specific_forward(self, x: Union[torch.Tensor, list], start_cell=-1, end_cell=None) -> [torch.Tensor]:
        """
        can execute specific part of the network,
        returns result after end_cell
        """
        raise NotImplementedError

    @use_eval
    def profile_macs(self, *inputs, batch_size=2) -> np.int64:
        """
        measure the required macs (memory access costs) of a forward pass
        prevent randomly changing the architecture
        """
        with torch.no_grad():
            if len(inputs) == 0:
                inputs = self.get_shape_in().random_tensor(batch_size=batch_size).to(self.get_device())
            if isinstance(inputs, (tuple, list)) and len(inputs) == 1:
                inputs = inputs[0]
            return torchprofile.profile_macs(self, args=inputs) // batch_size

    def is_external(self) -> bool:
        return Register.get_my_kwargs(self.__class__).get('external')
