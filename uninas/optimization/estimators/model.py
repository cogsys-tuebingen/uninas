import numpy as np
from uninas.models.abstract import AbstractModel
from uninas.methods.strategy_manager import StrategyManager
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
        self.model = self.model.prepare_predict(self._parsed_argument('model_device', args, index=index))
        self._cast_oh = self._parsed_argument('cast_one_hot', args, index=index)
        self._space = StrategyManager().get_value_space(unique=True)
        assert (not self._cast_oh) or (self._space.num_choices() > 0),\
            "If casting architecture tuples to one-hot, the strategy-manager value space must exist"

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        """ list arguments to add to argparse when this class (or a child class) is chosen """
        return super().args_to_add(index) + [
            Argument('model_file_path', default='{path_profiled}/model.pt', type=str, help='path to the model file'),
            Argument('model_device', default='', type=str, help='try to place the model on this device, if not empty (e.g. "cpu", "cuda:0")'),
            Argument('cast_one_hot', default='False', type=str, help='cast the tuple to one-hot', is_bool=True),
        ]

    def _evaluate_batch(self, x: np.array, strategy_name: str = None) -> np.array:
        """
        evaluate a batch of values at once
        :param x: [batch, ...] numpy array
        :param strategy_name: None if candidate is global, otherwise specific to this weight strategy
        :return: [batch, value]
        """
        assert strategy_name is None, "this estimator does not support specific strategy settings yet"
        if self._cast_oh:
            x = np.array([self._space.as_one_hot(v) for v in x])
        return self.model.predict(x)
