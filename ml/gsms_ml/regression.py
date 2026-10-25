"""Linear regression on the derived risk score (0-100).

The dataset's ``risk_score`` column is a transparent heuristic computed from
the keyword-classifier threat level plus corroboration/breaking/sentiment
bonuses (see ``fetch_dataset.level_risk_score``). This module fits ordinary
least squares — via sklearn's ``LinearRegression`` — to predict that score
from the seven standardized numeric features, and reports the regression
suite (MAE, MSE, RMSE, R²) plus a fixed-size sample of predicted-vs-actual
pairs for plotting.

The ``preprocess.Standardizer`` is fitted on the training split only and
then applied to the test split, mirroring the classification pipeline.
"""

from __future__ import annotations

from typing import Sequence

from sklearn.linear_model import LinearRegression

from .preprocess import Standardizer


def run_regression(
    X_train: Sequence[Sequence[float]],
    y_train: Sequence[float],
    X_test: Sequence[Sequence[float]],
    y_test: Sequence[float],
    feature_names: Sequence[str],
    sample_size: int = 50,
) -> dict:
    """Fit LinearRegression on standardized train features; score on test.

    Returns the metric suite (MAE/MSE/RMSE/R² on the test split, train R² for
    the overfitting picture), the fitted coefficients mapped to feature
    names, and an evenly-spaced sample (up to ``sample_size`` points) of
    predicted-vs-actual pairs aligned with the test split for plotting.
    """
    X_train_list = [list(row) for row in X_train]
    X_test_list = [list(row) for row in X_test]
    y_train_list = [float(value) for value in y_train]
    y_test_list = [float(value) for value in y_test]

    standardizer = Standardizer().fit(X_train_list)
    X_train_std = standardizer.transform(X_train_list)
    X_test_std = standardizer.transform(X_test_list)

    model = LinearRegression()
    model.fit(X_train_std, y_train_list)
    train_predictions = list(model.predict(X_train_std))
    test_predictions = [float(p) for p in model.predict(X_test_std)]

    n_test = len(y_test_list)
    mae = sum(abs(p - a) for p, a in zip(test_predictions, y_test_list)) / n_test
    mse = sum((p - a) ** 2 for p, a in zip(test_predictions, y_test_list)) / n_test

    def _r2(actual: list[float], predicted: list[float]) -> float:
        mean_actual = sum(actual) / len(actual)
        ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
        ss_tot = sum((a - mean_actual) ** 2 for a in actual)
        return 1.0 - ss_res / ss_tot if ss_tot else 0.0

    # evenly-spaced sample of test points for the predicted-vs-actual plot
    if n_test <= sample_size:
        indices = list(range(n_test))
    else:
        step = n_test / sample_size
        indices = [min(int(i * step), n_test - 1) for i in range(sample_size)]
    sample_predictions = [
        {
            "test_index": index,
            "actual": round(y_test_list[index], 3),
            "predicted": round(test_predictions[index], 3),
        }
        for index in indices
    ]

    return {
        "model": "LinearRegression",
        "target": "risk_score",
        "n_train": len(y_train_list),
        "n_test": n_test,
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(mse ** 0.5, 4),
        "r2": round(_r2(y_test_list, test_predictions), 4),
        "train_r2": round(_r2(y_train_list, train_predictions), 4),
        "mean_actual": round(sum(y_test_list) / n_test, 4),
        "coefficients": {
            name: round(float(coef), 4)
            for name, coef in zip(feature_names, model.coef_)
        },
        "intercept": round(float(model.intercept_), 4),
        "sample_predictions": sample_predictions,
    }
