import os
import numpy as np
import torch
from uninas.models.networks.uninas.search import SearchUninasNetwork
from uninas.models.networks.uninas.retrain import RetrainUninasNetwork
from uninas.methods.strategy_manager import StrategyManager
from uninas.optimization.profilers.abstract import AbstractProfiler
from uninas.optimization.profilers.functions import AbstractProfileFunction
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.utils.loggers.python import LoggerManager
from uninas.utils.args import Argument
from uninas.register import Register
from uninas.builder import Builder


@Register.profiler()
class SampleArchitecturesProfiler(AbstractProfiler):
    """
    sample architectures and save the profiled values in a way that a ProfiledData dataset can be used to learn them
    """

    def __init__(self, profile_fun: AbstractProfileFunction = None, is_test_run=False,
                 sample_overcomplete=True, sample_standalone=True,
                 num_train=10, num_test=10, **__):
        super().__init__(profile_fun, is_test_run, **__)
        self.sample_overcomplete = sample_overcomplete
        self.sample_standalone = sample_standalone
        self.num_train = num_train
        self.num_test = num_test

        self.logger = LoggerManager().get_logger()
        self.sizes = None
        self.sampled_overcomplete = {}
        self.sampled_standalone = {}

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('sample_overcomplete', default="True", type=str, help='sample from overcomplete models', is_bool=True),
            Argument('sample_standalone', default="True", type=str, help='sample from standalone models', is_bool=True),
            Argument('num_train', default=10, type=int, help='number of samples for the training set later'),
            Argument('num_test', default=10, type=int, help='number of samples for the test set later'),
        ]

    def _save(self, dir_: str):
        """ save the profiling data in this dir """
        def maybe_save(dct: dict, name: str):
            if len(dct) > 0:
                path = "%s/%s.pt" % (dir_, name)
                keys = list(dct.keys())
                train = {k: dct.get(k) for k in keys[:-self.num_test]}
                test = {k: dct.get(k) for k in keys[-self.num_test:]}
                torch.save(dict(sizes=self.sizes, train=train, test=test), path)
                self.logger.info("Saved %s data to %s" % (name, path))

        maybe_save(self.sampled_overcomplete, "overcomplete")
        maybe_save(self.sampled_standalone, "standalone")

    def _load(self, dir_: str):
        """ load the profiling data from this dir """
        def maybe_load(dct: dict, name: str):
            path = "%s/%s.pt" % (dir_, name)
            if os.path.isfile(path):
                data = torch.load(path)
                assert (data.get('sizes') == self.sizes) or (self.sizes is None)
                dct.update(data.get('train'))
                dct.update(data.get('test'))
                self.logger.info("Loaded %s data from %s" % (name, path))

        maybe_load(self.sampled_overcomplete, "overcomplete")
        maybe_load(self.sampled_standalone, "standalone")

    def profile(self, network: SearchUninasNetwork, mover: AbstractDeviceMover, batch_size: int):
        """ profile the network """
        sm = StrategyManager()
        network.set_forward_strategy(False)
        total_num = self.num_train + self.num_test

        # number of choices at every position
        max_choices = tuple(sm.get_num_choices())
        if self.sizes is not None:
            assert self.sizes == max_choices, "Mismatch between saved sizes (%s) and current ones (%s)" %\
                                              (self.sizes, max_choices)
        self.sizes = max_choices
        total_choices = int(np.prod(self.sizes))
        self.logger.info("max choices: %s" % repr(max_choices))
        assert total_choices >= total_num, "There are only %d architectures possible, but %d requested" %\
                                           (total_choices, total_num)

        # get the search space, we can sample random architectures from it
        space = sm.get_value_space(unique=True)

        # find out the size of the network inputs
        shape_in = network.get_shape_in()

        while True:
            if len(self.sampled_standalone) >= total_num or len(self.sampled_overcomplete) >= total_num:
                break

            # randomly sample a new architecture
            arc = space.random_sample()
            if arc in self.sampled_overcomplete or arc in self.sampled_standalone:
                continue

            # fix the network architecture
            sm.forward(fixed_arc=arc)

            if self.sample_overcomplete:
                # profile the value in an over-complete network, this is easy
                value = self.profile_fun.profile(module=network, shape_in=shape_in, mover=mover, batch_size=batch_size)
                self.sampled_overcomplete[arc] = value

            if self.sample_standalone:
                # profile the value in a stand-alone network:
                # - get the current network architecture (the last set fixed_arc indices will be used now)
                # - build it stand-alone (exactly as the "true" network would be used later), with the same input/output sizes
                # - place it on the profiled device
                # - profile that instead
                network_config = network.config(finalize=True)
                network_body = Builder().from_config(network_config)
                standalone = RetrainUninasNetwork(model_name='__tmp__', net=network_body, checkpoint_path='', assert_output_match=True)
                standalone.build(network.get_shape_in(), network.get_shape_out()[0])
                standalone = mover.move_module(standalone)
                value = self.profile_fun.profile(module=standalone, shape_in=shape_in, mover=mover, batch_size=batch_size)
                self.sampled_standalone[arc] = value

        self.logger.info("Sampled a total of %d architectures" % total_num)

        # unfix the network architecture again
        network.set_forward_strategy(True)
