import numpy as np
from uninas.models.abstract import AbstractModel
from uninas.methods.strategies.manager import StrategyManager
from uninas.optimization.estimators.abstract import AbstractEstimator
from uninas.utils.args import Argument, Namespace
from uninas.register import Register


@Register.hpo_estimator()
class ModelEstimator(AbstractEstimator):
    """
    An Estimator that uses a model for its predictions
    """

    def __init__(self, args: Namespace, index=None, **kwargs):
        super().__init__(args, index, **kwargs)
        self.model = AbstractModel.load_from(self._parsed_argument('model_file_path', args, index=index))
        self._cast_oh = self._parsed_argument('cast_one_hot', args, index=index)
        self._space = StrategyManager().get_value_space(unique=True)
        assert (not self._cast_oh) or (self._space.num_choices() > 0),\
            "If casting architecture tuples to one-hot, the strategy-manager value space must exist"

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('model_file_path', default='{path_profiled}/model.pt', type=str, help='path to the model file'),
            Argument('cast_one_hot', default='False', type=str, help='cast the tuple to one-hot', is_bool=True),
        ]

    def evaluate_tuple(self, values: tuple, strategy_name: str = None) -> float:
        """
        :param values: architecture description
        :param strategy_name: None if candidate is global, otherwise specific to this weight strategy
        """
        assert strategy_name is None, "this estimator does not support specific strategy settings yet"
        if self._cast_oh:
            values = self._space.as_one_hot(values)
        return self.model.predict(np.array([values]))[0]
