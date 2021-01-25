from uninas.networks.uninas.search import SearchUninasNetwork
from uninas.networks.uninas.retrain import RetrainUninasNetwork
from uninas.methods.strategies.manager import StrategyManager
from uninas.optimization.common.profilers.abstract import AbstractProfiler
from uninas.optimization.common.profilers.functions import AbstractProfileFunction
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.utils.args import Argument
from uninas.register import Register
from uninas.builder import Builder


@Register.profiler()
class NetworkSurrogateProfiler(AbstractProfiler):
    """
    use a neural network
    """

    def __init__(self, profile_fun: AbstractProfileFunction = None, is_test_run=False, example_argument=0, **__):
        super().__init__(profile_fun, is_test_run, **__)

        # currently the example argument is created in the 'args_to_add' method, and automatically given to this
        # __init__ function when the object is created (the names must match exactly)
        print('example argument', example_argument)

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('example_argument', default=10, type=int, help='just an example'),

            # e.g.
            # - desired data set size
            # - path to generated data set (check the nice is_path param in Argument), maybe continue if possible
            # - whether to build a sampled network as stand-alone or not

            # these parameters can be set in the config (e.g. experiments/examples/example_profile.py), like this:
            # "{cls_profiler}.example_argument": 20,
        ]

    def profile(self, network: SearchUninasNetwork, mover: AbstractDeviceMover, batch_size: int):
        """ profile the network """
        assert self.profile_fun is not None, "Can not measure if there is no profile function!"
        sm = StrategyManager()

        # step 1) generate a dataset
        # at some point, if other predictors are attempted (nearest neighbor, SVM, ...) step1 code could be moved
        # to a shared parent class

        # number of choices at every position
        max_choices = sm.get_num_choices()
        print("max choices", max_choices)

        # get the search space, we can sample random architectures from it
        space = sm.get_value_space(unique=True)
        for i in range(10):
            print("random arc %d: %s" % (i, space.random_sample()))

        # make sure that a forward pass will not change the network topology
        network.set_forward_strategy(False)

        # find out the size of the network inputs
        shape_in = network.get_shape_in()

        # fix the network architecture, profile it
        sm.forward(fixed_arc=space.random_sample())
        value = self.profile_fun.profile(module=network, shape_in=shape_in, mover=mover, batch_size=batch_size)
        print('value 1', value)

        # alternate way: instead of using one over-complete network that has unused modules,
        # - get the current network architecture (the last set fixed_arc indices will be used now)
        # - build it stand-alone (exactly as the "true" network would be used later), with the same input/output sizes
        # - place it on the profiled device
        # - profile that instead
        # this takes longer, but the mismatch between over-complete and stand-alone is very interesting to explore
        # can make this an option via Argument
        network_config = network.config(finalize=True)
        network_body = Builder().from_config(network_config)
        standalone = RetrainUninasNetwork(model_name='__tmp__', net=network_body, checkpoint_path='', assert_output_match=True)
        standalone.build(network.get_shape_in(), network.get_shape_out()[0])
        standalone = mover.move_module(standalone)
        value = self.profile_fun.profile(module=standalone, shape_in=shape_in, mover=mover, batch_size=batch_size)
        print('value 2', value)

        # save the data, generate a dataset... and be able to restore it, so step 1 can be skipped later

        # step 2) train the predictor

        # we will probably make a regular dataset, and train it the regular way - then almost everything as available
        # it should not affect anything from step1

        # other forms of prediction, e.g. SVM, can probably be fitted immediately

    def predict(self, values: tuple) -> float:
        """ predict the network's profile value with the given architecture """
        # for a giver architecture (tuple of ints), estimate the profiled value
        raise NotImplementedError
