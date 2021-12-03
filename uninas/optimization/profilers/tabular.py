import os
import torch
from uninas.models.networks.uninas.search import SearchUninasNetwork
from uninas.models.other.tabular import TabularSumModel
from uninas.methods.strategy_manager import StrategyManager
from uninas.optimization.profilers.abstract import AbstractProfiler
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.register import Register


@Register.profiler()
class TabularCellsProfiler(AbstractProfiler):
    """
    generate a look-up table, to know which module at which position has what profiled value (in isolation)
    this requires each cell to correspond to one unique architecture value
    (-> no shared architecture or cells without architecture weights)
    """

    def __init__(self, *_, **__):
        super().__init__(*_, *__)
        self.measured = {}

    def _get_measured_file(self, dir_: str) -> str:
        return "%s/%s.measured.pt" % (dir_, self.str())

    def _get_model_file(self, dir_: str) -> str:
        return "%s/%s.model.pt" % (dir_, self.str())

    def _save(self, dir_: str):
        """ save the profiling data in this dir """
        torch.save(self.measured, self._get_measured_file(dir_))
        model = TabularSumModel(table=self.measured.get('cells', {}),
                                constant=self.measured.get('stem', 0)+self.measured.get('head', 0))
        model.save(self._get_model_file(dir_))

    def _load(self, dir_: str):
        """ load the profiling data from this dir """
        path = self._get_measured_file(dir_)
        if os.path.isfile(path):
            self.measured.update(torch.load(path))

    def profile(self, network: SearchUninasNetwork, mover: AbstractDeviceMover, batch_size: int):
        """ profile the network """

        # unnecessary here, could check if this is a test and shorten everything
        # is_test_run = self.get('is_test_run')

        # stem
        if self.measured.get('stem', None) is None:
            self.logger.info('Measuring the stem')
            stem = network.get_stem()
            self.measured['stem'] = self.profile_fun.profile(stem, stem.get_shape_in(), mover, batch_size)

        # cells
        self.measured['cells'] = self.measured.get('cells', {})
        sm = StrategyManager()
        cells = network.get_cells()
        n_choices = sm.get_num_choices()
        if len(cells) != len(n_choices):
            raise ValueError("Number of cells (%d) must match number of arc choices (%d)" % (len(cells), len(n_choices)))
        network.set_forward_strategy(False)
        for i1, (cell, n) in enumerate(zip(cells, n_choices)):
            self.measured['cells'][i1] = self.measured['cells'].get(i1, {})
            for i2 in range(n):
                if self.measured['cells'][i1].get(i2, None) is None:
                    self.logger.info('Measuring cell %d, option %d' % (i1, i2))
                    sm.forward_const(i2)
                    self.measured['cells'][i1][i2] =\
                        self.profile_fun.profile(cell, cell.get_shape_in(), mover, batch_size)

        # final head
        if self.measured.get('head', None) is None:
            self.logger.info('Measuring the final head')
            head = network.get_heads()[-1]
            self.measured['head'] =\
                self.profile_fun.profile(head, head.get_shape_in(), mover, batch_size)
