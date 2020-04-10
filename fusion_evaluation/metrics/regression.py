from typing import List
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class RegressionEval:
    """Creates"""
    y_pred: List
    y_true: List
    num_features: int


@dataclass
class RegressionMetrics:
    """Creates a class for Regression Metrics

      Attributes:
        mean_squared_error (float): Mean squared error regression loss
        mean_absolute_error (float) :  Mean absolute error regression loss
        r_squared (float) :  Coefficient of determination
        r_square_adjusted (float): Adjusted R Squared based on number of features
    """
    mean_squared_error: float
    mean_absolute_error: float
    r_squared: float
    r_square_adjusted: float


def _r_square_adjusted(r_squared, y_pred, num_features):
    """Calculates adjusted r squared"""
    return (1 - r_squared) * ((len(y_pred) - 1) / (len(y_pred) - num_features))


def metrics(regression_eval: RegressionEval):
    """Returns metrics for prediction models """
    y_pred = regression_eval.y_pred
    y_true = regression_eval.y_true
    num_features = regression_eval.num_features

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    r_square_adj = _r_square_adjusted(r_squared, y_pred, num_features)
    return RegressionMetrics(mse, mae, r_squared, r_square_adj)
