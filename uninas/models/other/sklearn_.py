from uninas.models.abstract import AbstractWrapperModel
from uninas.utils.args import Argument
from uninas.register import Register


try:
    import sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


    @Register.model(can_fit=True, regression=True)
    class LinearRegressionSklearnModel(AbstractWrapperModel):
        """
        scikit-learn linear regression
        """
        _model_cls = LinearRegression

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return [
                Argument('fit_intercept', default="True", type=str, help="use no intercept if False, expecting the data to be centered", is_bool=True),
            ] + super().args_to_add(index)


    @Register.model(can_fit=True, regression=True)
    class SVMRegressionSklearnModel(AbstractWrapperModel):
        """
        scikit-learn support vector machine for regression
        """
        _model_cls = sklearn.svm.SVR

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return [
                Argument('kernel', default="rbf", type=str, choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], help='kernel for the SVM'),
                Argument('C', default=1.0, type=float, help='regularization parameter'),
                Argument('tol', default=1e-3, type=float, help='tolerance for the stopping criterion'),
                Argument('cache_size', default=512, type=float, help='kernel cache size in MB'),
                Argument('max_iter', default=-1, type=int, help='max iterations for the solver, no limit for -1'),
            ] + super().args_to_add(index)


    @Register.model(can_fit=True, classification=True)
    class LinearSVCSklearnModel(AbstractWrapperModel):
        """
        scikit-learn support vector machine for classification
        """
        _model_cls = sklearn.svm.LinearSVC

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return [
                Argument('penalty', default="l2", type=str, choices=['l1', 'l2'], help='penalty norm'),
                Argument('loss', default="squared_hinge", type=str, choices=['hinge', 'squared_hinge'], help='loss fuction'),
                Argument('dual', default="True", type=str, help='dual or primal optimizations problem', is_bool=True),
                Argument('C', default=1.0, type=float, help='regularization parameter'),
                Argument('tol', default=1e-3, type=float, help='tolerance for the stopping criterion'),
            ] + super().args_to_add(index)


    @Register.model(can_fit=True, regression=True)
    class RandomForestRegressorSklearnModel(AbstractWrapperModel):
        """
        scikit-learn random forest regressor
        """
        _none_args = ["max_depth", "max_samples"]
        _model_cls = RandomForestRegressor

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return [
                Argument('n_estimators', default=100, type=int, help='number of trees in the forest'),
                Argument('criterion', default="mse", type=str, choices=["mse", "mae"], help='fitting criterion'),
                Argument('max_depth', default=-1, type=int, help='max depth of the tree'),
                Argument('max_samples', default=-1, type=int, help='max samples for each tree'),
            ] + super().args_to_add(index)


    @Register.model(can_fit=True, classification=True)
    class RandomForestClassifierSklearnModel(AbstractWrapperModel):
        """
        scikit-learn random forest classifier
        """
        _none_args = ["max_depth"]
        _model_cls = RandomForestClassifier

        @classmethod
        def args_to_add(cls, index=None) -> [Argument]:
            """ list arguments to add to argparse when this class (or a child class) is chosen """
            return [
                Argument('n_estimators', default=100, type=int, help='number of trees in the forest'),
                Argument('criterion', default="gini", type=str, choices=["gini", "entropy"], help='fitting criterion'),
                Argument('max_depth', default=-1, type=int, help='max depth of the tree'),
            ] + super().args_to_add(index)


except ImportError as e:
    Register.missing_import(e)
