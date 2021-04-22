"""
this only exists to solve a cyclic dependency between Checkpointer and AbstractNetwork
"""


from typing import Union
import torch
import torch.nn as nn
from uninas.models.networks.abstract import AbstractNetwork
from uninas.training.callbacks.checkpoint import CheckpointCallback
from uninas.utils.args import Namespace
from uninas.utils.shape import Shape, ShapeList
from uninas.utils.torch.misc import count_parameters
from uninas.utils.loggers.python import LoggerManager, log_in_columns


class Abstract2Network(AbstractNetwork):

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> AbstractNetwork:
        """
        :param args: global argparse namespace
        :param index: argument index
        """
        raise NotImplementedError

    def _get_stem_output_shape(self) -> ShapeList:
        """ output shapes of the stem """
        training = self.training
        self.train(False)
        y = self.specific_forward(self.shape_in.random_tensor(batch_size=2), start_cell=-1, end_cell=-1)
        self.train(training)
        return ShapeList.from_tensors(y)

    def _get_cell_input_shapes(self) -> ShapeList:
        """ input shape(s) of each cell in order """
        shapes = ShapeList([self._get_stem_output_shape()])
        shapes.extend(self._get_cell_output_shapes())
        return shapes[:-1]

    def _get_cell_output_shapes(self) -> ShapeList:
        """ output shape(s) of each cell in order """
        training = self.training
        self.train(False)
        x = self.shape_in.random_tensor(batch_size=2)
        x = self.get_stem()(x)

        shapes = ShapeList([])
        for i in range(self.num_cells()):
            x = self.specific_forward(x, start_cell=i, end_cell=i)
            shapes.append(ShapeList.from_tensors(x))

        self.train(training)
        return shapes

    def _get_network_output_shapes(self) -> ShapeList:
        """ output shapes of the network """
        training = self.training
        self.train(False)
        cell_shapes = self.get_cell_output_shapes()
        y = self.specific_forward(self.get_heads_input_shapes()[-1].random_tensor(batch_size=2),
                                  start_cell=len(cell_shapes), end_cell=None)
        self.train(training)
        return ShapeList.from_tensors(y)

    def get_network(self) -> nn.Module:
        raise NotImplementedError

    def get_stem(self) -> nn.Module:
        raise NotImplementedError

    def get_cells(self) -> nn.ModuleList():
        raise NotImplementedError

    def get_heads(self) -> nn.ModuleList():
        raise NotImplementedError

    def _build(self, s_in: Shape, s_out: Shape) -> ShapeList:
        """ build the network, count params, log, maybe load pretrained weights """
        assert isinstance(s_out, Shape), "Attempting to build a network with an output that is not a Shape!"
        s_out_copy = s_out.copy(copy_id=True)
        self.shape_in = s_in.copy(copy_id=True)
        s_out_net = self._build2(s_in, s_out)
        LoggerManager().get_logger().info('Network built, it has %d parameters!' % self.get_num_parameters())

        # validate output shape sizes
        assert isinstance(s_out_net, ShapeList), "The network must output a list of Shapes, one shape per head! (ShapeList)"
        for shape in s_out_net.shapes:
            if not s_out_copy == shape:
                text = "One or more output shapes mismatch: %s, expected: %s" % (s_out_net, s_out_copy)
                if self.assert_output_match:
                    raise ValueError(text)
                else:
                    LoggerManager().get_logger().warning(text)
                    break

        # load weights?
        if len(self.checkpoint_path) > 0:
            path = CheckpointCallback.find_pretrained_weights_path(self.checkpoint_path, self.model_name,
                                                                   raise_missing=len(self.checkpoint_path) > 0)
            num_replacements = 1 if self.is_external() else 999
            self.loaded_weights(CheckpointCallback.load_network(path, self.get_network(), num_replacements))

        self.shape_out = s_out_net.shapes[0].copy(copy_id=True)
        self.shape_in_list = self.shape_in.shape
        self.shape_out_list = self.shape_out.shape
        return s_out_net

    def _build2(self, s_in: Shape, s_out: Shape) -> ShapeList:
        """ build the network """
        raise NotImplementedError

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


class AbstractExternalNetwork(Abstract2Network):

    def _build(self, s_in: Shape, s_out: Shape) -> ShapeList:
        """ build the network, count params, log, maybe load pretrained weights """
        s_in_net = s_in.copy(copy_id=True)
        super()._build(s_in, s_out)
        logger = LoggerManager().get_logger()
        rows = [('cell index', 'input shapes', 'output shapes', '#params'),
                ('stem', s_in.str(), self.get_stem_output_shape(), count_parameters(self.get_stem()))]
        logger.info('%s (%s):' % (self.__class__.__name__, self.model_name))
        for i, (s_in, s_out, cell) in enumerate(zip(self.get_cell_input_shapes(flatten=False),
                                                    self.get_cell_output_shapes(flatten=False), self.get_cells())):
            rows.append((i, s_in.str(), s_out.str(), count_parameters(cell)))
        rows.append(('head(s)', self.get_heads_input_shapes(), self.get_network_output_shapes(flatten=False),
                     count_parameters(self.get_heads())))
        rows.append(("complete network", s_in_net.str(), self.get_network_output_shapes(flatten=False),
                     count_parameters(self)))
        log_in_columns(logger, rows, start_space=4)
        return self.get_network_output_shapes(flatten=False)

    @classmethod
    def from_args(cls, args: Namespace, index=None) -> 'AbstractExternalNetwork':
        """
        :param args: global argparse namespace
        :param index: index for the arguments
        """
        all_parsed = cls._all_parsed_arguments(args, index=index)
        return cls(**all_parsed)

    def get_network(self) -> nn.Module:
        raise NotImplementedError

    def get_stem(self) -> nn.Module:
        raise NotImplementedError

    def get_cells(self) -> nn.ModuleList():
        raise NotImplementedError

    def get_heads(self) -> nn.ModuleList():
        raise NotImplementedError

    def _build2(self, s_in: Shape, s_out: Shape) -> ShapeList:
        """ build the network """
        raise NotImplementedError

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

    def config(self, **_) -> Union[None, dict]:
        return None
