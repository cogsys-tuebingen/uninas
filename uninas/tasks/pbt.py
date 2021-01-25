import os
import threading
import torch
from collections import defaultdict
from uninas.tasks.abstract import AbstractTask
from uninas.optimization.pbt.response import PbtServerResponse
from uninas.optimization.pbt.selector.abstract import AbstractPbtSelector
from uninas.utils.args import MetaArgument, Argument, Namespace
from uninas.utils.loggers.python import log_headline, log_in_columns
from uninas.utils.connection import get_ip
from uninas.register import Register
try:
    import Pyro5.api


    @Register.task()
    class PbtServerTask(AbstractTask):
        """
        A task that hosts a Population-based-training (PBT) server, where network training can connect to (via callback)
        """

        def __init__(self, args: Namespace, *args_, **kwargs):
            super().__init__(args, *args_, **kwargs)

            # necessities
            self.num_clients = self._parsed_argument('num_clients', args)
            self._lock = threading.Lock()
            self._first_use = True
            self._barrier_first = threading.Barrier(parties=self.num_clients, action=self._on_barrier_first_use)
            self._barrier = threading.Barrier(parties=self.num_clients, action=self._on_barrier_all_results)
            self._ip = get_ip()
            self._next_client_id = -1
            self._counter_done = 0
            self._epoch = -1
            self._next_callback_epoch = 0
            self._responses = {}
            self.weights_dir = '%s/weights/' % self.save_dir
            self.plots_dir = '%s/plots/' % self.save_dir
            for d in [self.weights_dir, self.plots_dir]:
                os.makedirs(d, exist_ok=True)

            # selector
            cls_selector = self._parsed_meta_argument(Register.pbt_selectors, 'cls_pbt_selector', args, index=None)
            self.selector = cls_selector.from_args(self.weights_dir, self.logger, args, index=None)
            assert isinstance(self.selector, AbstractPbtSelector)

            # data
            self._log_dicts = {}  # epoch, client_id, log_dict

            # pyro
            self._daemon = Pyro5.api.Daemon(host=self._ip)
            self._pyro_uri = self._daemon.register(self)

            # create the communications file
            self._communication_file = self._parsed_argument('communication_file', args)
            os.makedirs(os.path.dirname(self._communication_file), exist_ok=True)
            with open(self._communication_file, 'w+') as f:
                f.write(str(self._pyro_uri))

        @classmethod
        def _meta_path(cls, save_dir: str) -> str:
            return "%s/log_dicts.meta.pt" % save_dir

        @classmethod
        def meta_args_to_add(cls) -> [MetaArgument]:
            return [
                MetaArgument('cls_pbt_selector', Register.pbt_selectors, allowed_num=1,
                             help_name='selector to decide which checkpoints to keep and pursue'),
            ]

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return super().args_to_add(index) + [
                Argument('communication_file', default="{path_tmp}/communication_uri", type=str, is_path=True,
                         help="where the file to set up the first client-server communication is located"),
                Argument('num_clients', default=1, type=int, help="number of PBT clients"),
            ]

        @Pyro5.api.expose
        def client_register(self) -> PbtServerResponse:
            """ called by clients when they are created and want to figure out their id """
            try:
                with self._lock:
                    self._next_client_id += 1
                    assert self._next_client_id < self.num_clients, "Registering more clients than expected!"
                    self.logger.info("[{:<3}]   Register".format(self._next_client_id))
                    if self._next_client_id == self.num_clients - 1:
                        self.logger.info("All %d clients registered" % self.num_clients)
                    return PbtServerResponse(client_id=self._next_client_id)
            except Exception as e:
                self.logger.error("Server failure", exc_info=e)

        @Pyro5.api.expose
        def client_result(self, client_id: int, epoch: int, log_dict: dict) -> PbtServerResponse:
            try:
                # add client data
                with self._lock:
                    self._epoch = max(self._epoch, epoch)
                    self.logger.info('[{:<3}]   Result: epoch={:<6} dict={}'.format(client_id, epoch, str(log_dict)))
                    self._log_dicts[epoch] = self._log_dicts.get(epoch, defaultdict(dict))
                    log_dict.update(self._log_dicts[epoch][client_id])
                    self._log_dicts[epoch][client_id] = log_dict

                # first use, setting up initial parameters
                if self._first_use:
                    self._barrier_first.wait()
                    return self._responses[client_id]

                # if the watched key is not there, or we already synchronized, there is no need for synchronization
                with self._lock:
                    if (self._epoch < self._next_callback_epoch)\
                            or (not self.selector.is_interesting(epoch, self._log_dicts[epoch][client_id])):
                        return self.selector.empty_response(client_id)

                # synchronization, one at the barrier will figure out the new client instructions, then return them
                self._barrier.wait()
                self._barrier.reset()
                return self._responses[client_id]
            except Exception as e:
                self.logger.error("Server failure", exc_info=e)

        @Pyro5.api.expose
        def client_finish(self, client_id: int):
            """ called by clients when they finished their task """
            with self._lock:
                self.logger.info("[{:<3}]   Finished".format(client_id))
                self._counter_done += 1

        def _on_barrier_first_use(self):
            # let one client figure out client instructions
            log_headline(self.logger, 'Setting up initial mutations', target_len=80)
            self._first_use = False
            self._responses = {}
            lines = []
            for client_id, (log_dict, response) in self.selector.first_use(self._log_dicts[self._epoch]).items():
                self._responses[client_id] = response
                self._log_dicts[self._epoch][client_id].update(log_dict)
                lines.append(["[{:<3}]".format(client_id), "Mutated values", str(log_dict)])
            log_in_columns(self.logger, lines)
            log_headline(self.logger, 'Waiting for results', target_len=80)

        def _on_barrier_all_results(self):
            # let one client figure out client instructions
            log_headline(self.logger, 'Synchronizing', target_len=80)
            self._next_callback_epoch = self._epoch + 1
            torch.save(self._log_dicts, self._meta_path(self.save_dir))
            self.selector.save(self.save_dir)
            self._responses.clear()
            self._responses = self.selector.select(self._epoch, self._log_dicts[self._epoch])
            log_headline(self.logger, 'Waiting for results', target_len=80)

        def finish(self):
            log_headline(self.logger, 'Saving', target_len=80)
            self.selector.cleanup(self._epoch)
            self.selector.log_saves()
            self.selector.plot_results(self.plots_dir, self._log_dicts)
            self.logger.info('-'*80)

        def _loop_condition(self):
            with self._lock:
                if self._counter_done == self.num_clients:
                    self.logger.info("All %d clients reported completion" % self.num_clients)
                    self.finish()
                    return False
                return True

        def _load(self, checkpoint_dir: str) -> bool:
            """ load """
            meta = self._meta_path(checkpoint_dir)
            if os.path.isfile(meta):
                self._log_dicts = torch.load(meta)
                self.selector.load(checkpoint_dir)
                return True
            return False

        def _run(self):
            """ execute the task """
            log_headline(self.logger, "Hosting the PBT server")
            self.logger.info('{:<15} {:<}'.format('URI', str(self._pyro_uri)))
            self.logger.info('{:<15} {:<}'.format('URI file', self._communication_file))
            self._daemon.requestLoop(self._loop_condition)
            os.remove(self._communication_file)

except ImportError as e_:
    Register.missing_import(e_)
