from uninas.models.abstract import AbstractWrapperModel
from uninas.utils.args import Argument
from uninas.register import Register


try:
    import xgboost as xgb


    @Register.model(can_fit=True, regression=True)
    class RegressionXGBoostModel(AbstractWrapperModel):
        """
        gradient boosting regression model
        https://xgboost.ai
        """
        _none_args = ['max_depth']
        _model_cls = xgb.XGBRegressor

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return [
                Argument('max_depth', default=-1, type=int, help='maximum depth for base learners'),
                Argument('n_estimators', default=100, type=int, help='number of trees'),
                Argument('booster', default="dart", type=str, choices=['gbtree', 'gblinear', 'dart'], help='which booster to use'),
                Argument('objective', default="reg:squarederror", type=str, choices=["reg:linear", "reg:squarederror"], help='objective to minimize'),
            ] + super().args_to_add(index)


except ImportError as e:
    Register.missing_import(e)
