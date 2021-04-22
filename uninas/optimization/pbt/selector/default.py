import math
from uninas.optimization.target import OptimizationTarget
from uninas.optimization.pbt.mutations.abstract import AbstractPbtMutation
from uninas.optimization.pbt.save import PbtSave
from uninas.optimization.pbt.selector.abstract import AbstractPbtSelector
from uninas.optimization.pbt.response import PbtServerResponse
from uninas.utils.args import Argument
from uninas.utils.loggers.python import Logger
from uninas.utils.misc import flatten
from uninas.register import Register


@Register.pbt_selector()
class DefaultPbtSelector(AbstractPbtSelector):
    """
    default selector for PBT:
    - discards percentage of worst performing individuals per epoch,
    - replaces them from checkpoints of percentage of best performing ones
    """

    def __init__(self, weights_dir: str, logger: Logger, targets: [OptimizationTarget], mutations: [AbstractPbtMutation],
                 each_epochs: int, grace_epochs: int, save_clone: bool, elitist: bool,
                 replace_worst: float, copy_best: float):
        super().__init__(weights_dir, logger, targets, mutations, each_epochs, grace_epochs, save_clone, elitist)
        self.replace_worst = replace_worst
        self.copy_best = copy_best

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('replace_worst', default=0.3, type=float, help="percentage of worst performing individuals to replace"),
            Argument('copy_best', default=0.2, type=float, help="percentage of best performing individuals to copy"),
        ]

    def _select(self, responses: {int: PbtServerResponse}, epoch: int, log_dicts: {int: dict}) -> [(PbtSave, PbtSave)]:
        """
        create the responses for each client, {client_id: log_dict}, mark all saves that should be kept
        :param responses: {client_id: PbtServerResponse}
        :param epoch: current epoch
        :param log_dicts: {client_id: dict}
        :return replacements [(to_replace, replace_with)]
        """
        # get sorted list of best checkpoints
        sorted_cur = flatten(self.get_best(self.get_saves_list(), epoch=epoch, exclude_old=True))
        sorted_all = flatten(self.get_best(self.get_saves_list())) if self.elitist else sorted_cur

        # best (of any epoch if elitist) are kept
        idx_b = math.ceil(len(sorted_all) * self.copy_best)
        best = sorted_all[:idx_b]
        best_keys = [b.key for b in best]
        for save in best:
            save.add_usage()
            if save.epoch == epoch:
                responses[save.client_id].save_path = save.get_path()

        # worst of current epoch will be replaced, unless it also belongs to the best keys
        idx_w = math.floor(len(sorted_cur) * (1 - self.replace_worst))
        worst = sorted_cur[idx_w:]
        worst = [w for w in worst if w.key not in best_keys]
        replacements = []
        for i, to_replace in enumerate(worst):
            replacements.append((to_replace, best[i % len(best)]))

        return replacements
