from uninas.networks.self.search import SearchUninasNetwork
from uninas.methods.strategies.manager import StrategyManager
from uninas.optimization.common.profilers.abstract import AbstractProfiler
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.register import Register


@Register.profiler()
class TabularCellsProfiler(AbstractProfiler):
    """
    generate a look-up table, to know which module at which position has what profiled value (in isolation)
    this requires each cell to correspond to one unique architecture value
    (-> no shared architecture or cells without architecture weights)
    """

    def profile(self, network: SearchUninasNetwork, mover: AbstractDeviceMover, batch_size: int):
        """ profile the network """
        assert self.profile_fun is not None, "Can not measure if there is no profile function!"

        # unnecessary here, could check if this is a test and shorten everything
        # is_test_run = self.get('is_test_run')

        # set up nested structure if it does not exist
        self.data['measured'] = self.data.get('measured', {})

        # stem
        if self.data.get('measured').get('stem', None) is None:
            self.logger.info('Measuring the stem')
            stem = network.get_stem()
            self.data['measured']['stem'] =\
                self.profile_fun.profile(stem, stem.get_cached('shape_in'), mover, batch_size)

        # cells
        self.data['measured']['cells'] = self.data.get('measured').get('cells', {})
        sm = StrategyManager()
        cells = network.get_cells()
        n_choices = sm.ordered_num_choices()
        if len(cells) != len(n_choices):
            raise ValueError("Number of cells (%d) must match number of arc choices (%d)" % (len(cells), len(n_choices)))
        network.set_forward_strategy(False)
        for i1, (cell, n) in enumerate(zip(cells, n_choices)):
            self.data['measured']['cells'][i1] = self.data['measured']['cells'].get(i1, {})
            for i2 in range(n):
                if self.data['measured']['cells'][i1].get(i2, None) is None:
                    self.logger.info('Measuring cell %d, option %d' % (i1, i2))
                    sm.forward_const(i2)
                    self.data['measured']['cells'][i1][i2] =\
                        self.profile_fun.profile(cell, cell.get_cached('shape_in'), mover, batch_size)

        # final head
        if self.data.get('measured').get('head', None) is None:
            self.logger.info('Measuring the final head')
            head = network.get_heads()[-1]
            self.data['measured']['head'] =\
                self.profile_fun.profile(head, head.get_cached('shape_in'), mover, batch_size)

    def predict(self, values: tuple) -> float:
        """ predict the network's profile value with the given architecture """
        cells = [self.data.get('measured').get('cells').get(i).get(v) for i, v in enumerate(values)]
        return self.data.get('measured').get('stem') + self.data.get('measured').get('head') + sum(cells)
