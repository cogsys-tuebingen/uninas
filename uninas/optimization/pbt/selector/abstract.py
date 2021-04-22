from typing import Union, List
import numpy as np
import torch
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from uninas.optimization.target import OptimizationTarget
from uninas.optimization.pbt.mutations.abstract import AbstractPbtMutation
from uninas.optimization.pbt.save import PbtSave
from uninas.optimization.pbt.response import PbtServerResponse
from uninas.optimization.pbt.events import AbstractPbtEvent, ReplacementPbtEvent
from uninas.utils.args import ArgsInterface, MetaArgument, Argument, Namespace
from uninas.utils.loggers.python import Logger, log_in_columns
from uninas.utils.misc import flatten
from uninas.register import Register


class AbstractPbtSelector(ArgsInterface):
    """
    to figure out which clients should save, which should load from where, ...
    """

    def __init__(self, weights_dir: str, logger: Logger, targets: [OptimizationTarget], mutations: [AbstractPbtMutation],
                 each_epochs: int, grace_epochs: int, save_clone: bool, elitist: bool):
        super().__init__()
        self._nds = NonDominatedSorting()
        self._data = {
            'saves': dict(),         # {key: PbtSave}
            'replacements': [],      # ReplacementPbtEvent
        }
        self.weights_dir = weights_dir
        self.logger = logger
        self.targets = targets
        self.mutations = mutations
        self.each_epochs = each_epochs
        self.grace_epochs = grace_epochs
        self.save_clone = save_clone
        self.elitist = elitist

    @classmethod
    def meta_args_to_add(cls) -> [MetaArgument]:
        return [
            MetaArgument('cls_pbt_targets', Register.optimization_targets, allow_duplicates=True,
                         help_name='optimization target(s)'),
            MetaArgument('cls_pbt_mutations', Register.pbt_mutations,
                         help_name='mutations to the copied checkpoint training state'),
        ]

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('each_epochs', default=1, type=int, help="only synchronize each n epochs"),
            Argument('grace_epochs', default=0, type=int, help="skip synchronization for the first n epochs"),
            Argument('save_clone', default="True", type=str, is_bool=True,
                     help="save the clone model weights if available, otherwise save the trained model's weights"),
            Argument('elitist', default="True", type=str, is_bool=True,
                     help="elitist: keep old checkpoints if they are better"),
        ]

    @classmethod
    def from_args(cls, save_dir: str, logger: Logger, args: Namespace, index=None) -> 'AbstractPbtSelector':
        targets = [cls_target.from_args(args, index=i) for
                   i, cls_target in enumerate(cls._parsed_meta_arguments(Register.optimization_targets, 'cls_pbt_targets', args, index))]
        mutations = [cls_mutations.from_args(args, index=i) for
                     i, cls_mutations in enumerate(cls._parsed_meta_arguments(Register.pbt_mutations, 'cls_pbt_mutations', args, index))]
        return cls(save_dir, logger, targets, mutations, **cls._all_parsed_arguments(args, index=index))

    @classmethod
    def _meta_name(cls, save_dir: str) -> str:
        return "%s/%s.meta.pt" % (save_dir, cls.__name__)

    def save(self, save_dir: str):
        torch.save(self._data, self._meta_name(save_dir))

    def load(self, save_dir: str):
        self._data = torch.load(self._meta_name(save_dir))

    # keep track of saves/checkpoints

    def get_saves(self) -> {str: PbtSave}:
        return self._data['saves']

    def get_saves_list(self) -> [PbtSave]:
        return list(self.get_saves().values())

    def get_save(self, epoch: int, client_id: int) -> Union[None, PbtSave]:
        for save in self.get_saves_list():
            if save.epoch == epoch and save.client_id == client_id:
                return save

    def save_client(self, epoch: int, client_id: int, log_dict: dict) -> PbtSave:
        """ save """
        path = "%s/%d-%d-weights.pt" % (self.weights_dir, epoch, client_id)
        save = PbtSave(epoch, client_id, log_dict, path=path)
        self._data['saves'][save.key] = save
        return save

    def remove_saves_by_keys(self, keys: list):
        """ specific saves """
        if len(keys) > 0:
            self.logger.info("Removing saves:")
            saves = [self.get_saves()[k] for k in keys]
            self.log_saves(saves)
            for s in saves:
                s.remove_file()
                del self._data['saves'][s.key]

    def remove_unused_saves(self):
        """ remove all saves that are not marked as used """
        self.remove_saves_by_keys([save.key for save in self.get_saves_list() if not save.is_used()])

    def get_best(self, saves: [PbtSave], epoch: int = None, exclude_old=False) -> List[List[PbtSave]]:
        """
        get the saves in order from best to worst, optionally of a specific epoch
        :param saves
        :param epoch: prefer checkpoints of this epochs, append the others at the back
        :param exclude_old: do not append any old checkpoints, only functions is epoch is set
        """
        # get the ranking
        log_dicts = [c.log_dict for c in saves]
        values = np.array([[target.sort_value(ld) for target in self.targets] for ld in log_dicts])
        _, rank = self._nds.do(values, return_rank=True)
        # sort the saves, possibly older ones to the back
        best = []
        old_best = []
        for i in range(max(rank)+1):
            # rank i
            best_ = []
            for c, r in zip(saves, rank):
                if r == i:
                    best_.append(c)
            # old ones separated?
            if isinstance(epoch, int):
                best.append([s for s in best_ if s.epoch == epoch])
                if not exclude_old:
                    old_best.append([s for s in best_ if s.epoch != epoch])
            else:
                best.append(best_)
        # add old ones (maybe empty), remove empty lists, return
        best += old_best
        best = [b for b in best if len(b) > 0]
        return best

    def cleanup(self, epoch: int = None):
        """
        keep the best saves (pareto front), remove the rest
        :param epoch: if the selector is not elitist, remove older saves regardless of performance
        """
        best = self.get_best(self.get_saves_list(), epoch=None if self.elitist else epoch)
        keys = []
        for i, rank_i in enumerate(best):
            if i == 0:
                continue
            keys.extend([c.key for c in rank_i])
        self.remove_saves_by_keys(keys)
        assert len(self.get_saves()) > 0

    def log_saves(self, saves: [PbtSave] = None):
        if saves is None:
            self.logger.info("All saves:")
            saves = flatten(self.get_best(self.get_saves_list()))
        saves = sorted(saves, key=lambda save: self.targets[0].sort_value(save.log_dict))
        lines = [["epoch=%d" % save.epoch]
                 + ["client=%d" % save.client_id]
                 + [target.as_str(save.log_dict) for target in self.targets]
                 + [save.get_path() if save.is_used() else ""]
                 for save in saves]
        log_in_columns(self.logger, lines, add_bullets=True)

    # keep track of events

    def add_replacement_event(self, r: ReplacementPbtEvent):
        self._data['replacements'].append(r)

    def get_replacement_events(self, epoch: int = None, client_id: int = None):
        """
        get list of all replacements that happened, optionally filter
        :param epoch:
        :param client_id:
        """
        events = self._data['replacements']
        if isinstance(epoch, int):
            events = [e for e in events if e.epoch == epoch]
        if isinstance(client_id, int):
            events = [e for e in events if e.client_id == client_id]
        return events

    def log_events(self, events: [AbstractPbtEvent], text: str = None):
        if len(events) > 0:
            if isinstance(text, str):
                self.logger.info(text)
            lines = [e.str_columns() for e in events]
            log_in_columns(self.logger, lines, add_bullets=True)

    # selecting, mutating, responses

    @classmethod
    def empty_response(cls, client_id: int) -> PbtServerResponse:
        return PbtServerResponse(client_id=client_id)

    def is_interesting(self, epoch: int, log_dict: dict) -> bool:
        """ if the watched key is not in the log dict, no need to synchronize """
        if epoch < self.grace_epochs:
            return False
        if (epoch - self.grace_epochs + 1) % self.each_epochs > 0:
            return False
        return all([target.is_interesting(log_dict) for target in self.targets])

    def first_use(self, log_dicts: {int, dict}) -> {int, (dict, PbtServerResponse)}:
        """ create the changed log dict and response for each client """
        ret = {}
        for client_id, log_dict in log_dicts.items():
            ld, r = log_dict, self.empty_response(client_id)
            for m in self.mutations:
                ld, r = m.initial_mutate(r, ld, len(log_dicts))
            ret[client_id] = (ld, r)
        return ret

    def select(self, epoch: int, log_dicts: {int, dict}) -> {int, PbtServerResponse}:
        """ create the responses for each client """
        # reset mutations
        for m in self.mutations:
            m.reset()
        # reset usage of all saves
        for save in self.get_saves_list():
            save.reset()
        # add all to saves, without actually saving states yet (not necessary for all)
        for client_id, log_dict in log_dicts.items():
            self.save_client(epoch, client_id, log_dict)
        # empty responses
        responses = {}
        for client_id, log_dict in log_dicts.items():
            responses[client_id] = PbtServerResponse(client_id=client_id, save_clone=self.save_clone)
        # get replacements
        for (replace, replace_with) in self._select(responses, epoch, log_dicts):
            replace_with.add_usage()
            e = ReplacementPbtEvent(replace.epoch, replace.client_id,
                                    replace_with.epoch, replace_with.client_id, replace_with.get_path())
            self.add_replacement_event(e)
            responses[replace.client_id].load_path = replace_with.get_path()
            # apply mutations
            for m in self.mutations:
                responses[replace.client_id] = m.mutate(responses[replace.client_id], replace_with.log_dict)
        # remove all unused saves
        self.remove_unused_saves()
        # log replacements
        self.log_events(self.get_replacement_events(epoch=epoch), text="Replacements:")
        # log remaining saves, return responses
        self.log_saves()
        return responses

    def _select(self, responses: {int: PbtServerResponse}, epoch: int, log_dicts: {int: dict}) -> [(PbtSave, PbtSave)]:
        """
        create the responses for each client, {client_id: log_dict}, mark all saves that should be kept
        :param responses: {client_id: PbtServerResponse}
        :param epoch: current epoch
        :param log_dicts: {client_id: dict}
        :return replacements [(to_replace, replace_with)]
        """
        raise NotImplementedError

    # plotting results

    def plot_results(self, save_dir: str, log_dicts: dict):
        """ task complete, plot results, {epoch: {client_id: log_dict}} """
        # reshape into {key: {client_id: list of entries}}
        epochs = sorted(list(log_dicts.keys()))
        clients = list(log_dicts[epochs[0]].keys())
        keys = log_dicts[epochs[-1]][clients[0]]
        reshaped_log_dicts = {k: {c: list() for c in clients} for k in keys}
        for e in epochs:
            for c in clients:
                for k in keys:
                    reshaped_log_dicts[k][c].append(log_dicts[e][c].get(k, None))
        # actually plot
        self._plot_results(save_dir, log_dicts, reshaped_log_dicts, epochs, clients, keys)

    def _plot_results(self, save_dir: str, log_dicts: dict, reshaped_log_dicts: dict,
                      epochs: [int], clients: [int], keys: [str]):
        """

        :param save_dir: where to save
        :param log_dicts: {epoch: {client_id: {key: value}}}
        :param reshaped_log_dicts: {key: {client_id: list of values}}
        :param epochs: list of epochs
        :param clients: list of client ids
        :param keys: list of keys in the log dicts
        """
        # plot all by keys
        for key in keys:
            is_id_relevant = True
            for n in ['train', 'val', 'test']:
                if key.startswith(n):
                    is_id_relevant = False
                    break
            fmt = 'o--' if is_id_relevant else '-'
            for c in clients:
                plt.plot(epochs, reshaped_log_dicts[key][c], fmt, label="%d" % c)
            if is_id_relevant:
                plt.legend()
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.savefig("%s/key_%s.pdf" % (save_dir, key.replace('/', '_')))
            plt.clf()

        # lineage
        for e0, e1 in zip(epochs[:-1], epochs[1:]):
            for client_id in clients:
                if e0 == 0:
                    plt.plot((e0, e0), (client_id, client_id), 'bo-', linewidth=1)
                r = self.get_replacement_events(epoch=e0, client_id=client_id)
                if len(r) == 0:
                    # no replacement happened
                    plt.plot((e0, e1), (client_id, client_id), 'bo-', linewidth=1)
                else:
                    # no replacement happened
                    if len(r) > 1:
                        self.logger.warning("Have %d replacements for client_id=%d, epoch1=%d, only taking the first" %
                                            (len(r), client_id, e1))
                    r = r[0]
                    plt.plot((r.epoch_replaced_with, e1), (r.client_replaced_with, client_id), 'ro--', linewidth=2)
        plt.xlabel("epoch")
        plt.ylabel("client id")
        plt.savefig("%s/lineage.pdf" % save_dir)
        plt.clf()


if __name__ == '__main__':
    sel = AbstractPbtSelector('', Logger('__main__'), [], [], 4, 20, False, False)
    for i_ in range(100):
        if sel.is_interesting(i_, {}):
            print(i_, end=', ')
