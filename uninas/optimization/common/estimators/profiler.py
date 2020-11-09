from uninas.optimization.common.estimators.abstract import AbstractEstimator
from uninas.optimization.common.profilers.abstract import AbstractProfiler
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


@Register.hpo_estimator()
class ProfilerEstimator(AbstractEstimator):
    """
    An Estimator that uses a "trained" profiler for its predictions
    """

    def __init__(self, args: Namespace, index=None, **kwargs):
        super().__init__(args, index, **kwargs)
        self.profiler = AbstractProfiler.from_file(self._parsed_argument('profiler_file_path', args, index=index))

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('load', default="False", type=str, help='load the cached weights or continue', is_bool=True),
            Argument('profiler_file_path', default='{path_tmp}/profile/TabularCellsProfiler.LatencyProfileFunction.pt',
                     type=str, help='path to the profiler save file'),
        ]

    def evaluate_tuple(self, values: tuple, strategy_name: str = None) -> float:
        """
        :param values: architecture description
        :param strategy_name: None if candidate is global, otherwise specific to this weight strategy
        """
        assert strategy_name is None, "this estimator does not support specific strategy settings yet"
        return self.profiler.predict(values)
