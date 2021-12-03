import os
import time
from uninas.methods.abstract_method import AbstractMethod
from uninas.training.trainer.abstract import AbstractTrainerFunctions
from uninas.optimization.pbt.response import PbtServerResponse
from uninas.training.callbacks.abstract import AbstractCallback
from uninas.utils.torch.misc import itemize
from uninas.utils.args import Argument
from uninas.utils.loggers.python import LoggerManager
from uninas.register import Register
try:
    import Pyro5.api


    @Register.training_callback(requires_log_dict=True)
    class PbtCallback(AbstractCallback):
        """
        Communicate with a Population-based-training (PBT) server for saving/loading/param instructions
        """

        def __init__(self, save_dir: str, index: int, communication_file: str):
            """
            :param save_dir: main dir where to save
            :param index: index of this callback
            :param communication_file: where the file to set up the first client-server communication is located
            """
            super().__init__(save_dir, index)
            self._communication_file = communication_file
            self._is_connected = False
            self._server_uri = None
            self._server = None
            self._client_id = -1

        @classmethod
        def log(cls, msg: str):
            LoggerManager().get_logger().info('%s: %s' % (cls.__name__, msg))

        def setup(self, trainer: AbstractTrainerFunctions, pl_module: AbstractMethod, stage: str):
            """ called when the trainer changes the method it trains (also called for the first one) """
            assert not self._is_connected, "Can not change the method"
            while not self._is_connected:
                time.sleep(1)
                if os.path.isfile(self._communication_file):
                    with open(self._communication_file, 'r') as f:
                        self._server_uri = f.read()
                        self._server = Pyro5.api.Proxy(self._server_uri)
                        self.log("connecting to URI: %s" % self._server_uri)
                        response = PbtServerResponse.from_dict(self._server.client_register())
                        self._client_id = response.client_id
                        self.log("local client id: %d" % self._client_id)
                        self._is_connected = True
                        self._on_server_response(response, trainer, pl_module)

        def teardown(self, trainer: AbstractTrainerFunctions, pl_module: AbstractMethod, stage: str):
            """Called when fit or test ends"""
            self._server.client_finish(self._client_id)
            del self._server
            self._is_connected = False

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return super().args_to_add(index) + [
                Argument('communication_file', default="{path_tmp}/communication_uri", type=str, is_path=True,
                         help="where the file to set up the first client-server communication is located"),
            ]

        def _on_server_response(self, response: PbtServerResponse, trainer: AbstractTrainerFunctions,
                                pl_module: AbstractMethod):
            assert self._client_id == response.client_id,\
                "client_id mismatch! Got %d, expected %d" % (response.client_id, self._client_id)
            response.act(self.log, trainer)

        def _client_result(self, log_dict: dict, trainer: AbstractTrainerFunctions, pl_module: AbstractMethod):
            assert isinstance(log_dict, dict)
            r = self._server.client_result(self._client_id, pl_module.current_epoch, itemize(log_dict))
            r = PbtServerResponse.from_dict(r)
            self._on_server_response(r, trainer, pl_module)

        def on_train_epoch_start(self, trainer: AbstractTrainerFunctions,
                                 pl_module: AbstractMethod,
                                 log_dict: dict = None):
            """ Called when the train epoch begins. """
            self._client_result(log_dict, trainer, pl_module)

        def on_train_epoch_end(self, trainer: AbstractTrainerFunctions,
                               pl_module: AbstractMethod,
                               log_dict: dict = None):
            """ Called when the train epoch ends. """
            self._client_result(log_dict, trainer, pl_module)

        def on_validation_epoch_end(self, trainer: AbstractTrainerFunctions,
                                    pl_module: AbstractMethod,
                                    log_dict: dict = None):
            """ Called when the val epoch ends. """
            self._client_result(log_dict, trainer, pl_module)

        def on_test_epoch_end(self, trainer: AbstractTrainerFunctions,
                              pl_module: AbstractMethod,
                              log_dict: dict = None):
            """ Called when the test epoch ends. """
            self._client_result(log_dict, trainer, pl_module)

except ImportError as e:
    Register.missing_import(e)
