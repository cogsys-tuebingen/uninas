import torch
from uninas.modules.modules.misc import SumParallelModules
from uninas.methods.strategy_manager import StrategyManager
from uninas.register import Register


@Register.network_mixed_op()
class MixedOp(SumParallelModules):
    """
    all op choices on one path in parallel,
    the weight strategy decides which results to compute and combine
    """

    def __init__(self, submodules: list, priors: list, name: str, strategy_name: str):
        """
        :param submodules: list or nn.ModuleList of choices
        :param priors: list of indices, which prior candidates to consider for additional super-network weights
        :param name: name of the architecture weight
        :param strategy_name: name of the architecture strategy to use
        """
        super().__init__(submodules)
        self._add_to_kwargs(name=name, strategy_name=strategy_name)
        self.sm = StrategyManager()
        self.ws = self.sm.make_weight(self.strategy_name, name, only_single_path=False, choices=self.submodules)

    def _get_prev_names(self, name: str, priors: [int], include_self=True) -> [str]:
        all_prev_names = self.sm.ordered_names(unique=False)[:-1]
        prev_names = []
        for i in priors:
            try:
                prev_names.append(all_prev_names[i])
            except:
                pass
        if include_self:
            prev_names.append(name)
        # print(name, self.__class__.__name__, prev_names)
        return prev_names

    def config(self, finalize=True, **_) -> dict:
        if finalize:
            indices = self.ws.get_finalized_indices(self.name)
            if len(indices) == 1:
                return self.submodules[indices[0]].config(finalize=finalize, **_)
            return SumParallelModules([self.submodules[i] for i in indices]).config(finalize=finalize, **_)
        else:
            return super().config(finalize=finalize, **_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ws.combine(self.name, x, self.submodules)


class AbstractDependentMixedOp(MixedOp):
    """
    a mixed op that somehow depends on previously chosen mixed ops
    """

    def _save_to_state_dict(self, destination: dict, prefix: str, keep_vars: bool):
        # save additional info
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination['%s@added_state' % prefix] = self._save_add_dict()

    def _load_from_state_dict(self, state_dict: dict, prefix: str, local_metadata, strict,
                              missing_keys: list, unexpected_keys: list, error_msgs: list):
        # load additional info
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        added_state = state_dict.get('%s@added_state' % prefix, {})
        if '%s@added_state' % prefix in unexpected_keys:
            unexpected_keys.remove('%s@added_state' % prefix)
        self._load_add_dict(added_state)

    def _save_add_dict(self) -> dict:
        """ additional info stored in the save_dict """
        return {}

    def _load_add_dict(self, dct: dict):
        """ additional info restored from the save_dict """
        pass
