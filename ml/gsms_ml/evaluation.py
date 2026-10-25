"""Evaluation toolbox: cross-validation, bootstrap CIs, bias/variance demos.

- ``cross_validate_accuracy``: stratified k-fold (shuffled, seeded) with a
  graceful fallback: if some class has fewer members than the requested k
  (common with the rare 'critical' band), the fold count shrinks to the
  smallest class count and the result records that.
- ``bootstrap_metric``: nonparametric bootstrap over test predictions —
  resample the test set with replacement, recompute the metric, and report
  the 2.5/97.5 percentile interval. Gives an honest uncertainty band on every
  headline number in the report.
- ``bias_variance_demo``: decision-tree depth sweep 1..12 showing the
  underfitting→overfitting arc (train error keeps falling, test error turns
  around).
- ``regularization_demo``: LogisticRegression C sweep showing how stronger
  regularization (small C) closes the train/test gap.

Every path is seeded: StratifiedKFold(random_state=42), bootstrap via
numpy's default_rng(42), and every model factory pins random_state=42.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .supervised import SEED

MetricFn = Callable[[Sequence, Sequence], float]
ModelFactory = Callable[[], object]


def cross_validate_accuracy(
    model_factory: ModelFactory,
    X: Sequence[Sequence[float]],
    y: Sequence,
    k: int = 5,
) -> dict:
    """Stratified k-fold accuracy for a model factory (shuffled, seeded).

    Returns per-fold accuracies plus their mean and standard deviation. If
    the rarest class cannot fill ``k`` folds, the fold count is reduced to
    that class count (>= 2) and the adjustment is recorded in ``notes``.
    """
    X_array = [list(row) for row in X]
    y_list = list(y)
    notes: list[str] = []

    smallest_class = min(Counter(y_list).values()) if y_list else 0
    n_splits = k
    if smallest_class < k:
        if smallest_class < 2:
            return {
                "k": k,
                "fold_accuracies": [],
                "mean": None,
                "std": None,
                "notes": ["smallest class has fewer than 2 members; CV skipped"],
            }
        n_splits = smallest_class
        notes.append(
            f"smallest class has {smallest_class} members; fold count reduced "
            f"from {k} to {n_splits}"
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_accuracies: list[float] = []
    for train_index, test_index in splitter.split(X_array, y_list):
        X_fold_train = [X_array[i] for i in train_index]
        X_fold_test = [X_array[i] for i in test_index]
        y_fold_train = [y_list[i] for i in train_index]
        y_fold_test = [y_list[i] for i in test_index]
        model = model_factory()
        model.fit(X_fold_train, y_fold_train)
        fold_accuracies.append(
            float(accuracy_score(y_fold_test, model.predict(X_fold_test)))
        )

    mean = float(np.mean(fold_accuracies)) if fold_accuracies else None
    std = float(np.std(fold_accuracies)) if fold_accuracies else None
    return {
        "k": n_splits,
        "fold_accuracies": [round(value, 4) for value in fold_accuracies],
        "mean": round(mean, 4) if mean is not None else None,
        "std": round(std, 4) if std is not None else None,
        "notes": notes,
    }


def bootstrap_metric(
    y_true: Sequence,
    y_pred: Sequence,
    metric_fn: MetricFn,
    n_boot: int = 200,
    seed: int = SEED,
) -> dict:
    """Percentile bootstrap (2.5/97.5) for any (y_true, y_pred) metric.

    Resamples the test predictions with replacement ``n_boot`` times using a
    seeded numpy Generator, recomputes ``metric_fn`` on each resample, and
    returns the point estimate, the bootstrap mean and the percentile
    interval. Resamples where the metric is undefined (e.g. a single class
    reaching a macro-F1 scorer) are skipped and counted.
    """
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    n = len(y_true_list)
    point = float(metric_fn(y_true_list, y_pred_list))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    skipped = 0
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        sample_true = [y_true_list[i] for i in indices]
        sample_pred = [y_pred_list[i] for i in indices]
        try:
            values.append(float(metric_fn(sample_true, sample_pred)))
        except (ValueError, ZeroDivisionError):
            skipped += 1
    if values:
        lo, hi = np.percentile(values, [2.5, 97.5])
        mean = float(np.mean(values))
    else:
        lo = hi = mean = point
    return {
        "point": round(point, 4),
        "mean": round(mean, 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "n_boot": n_boot,
        "skipped_resamples": skipped,
        "seed": seed,
    }


def bias_variance_demo(
    X_train: Sequence[Sequence[float]],
    y_train: Sequence,
    X_test: Sequence[Sequence[float]],
    y_test: Sequence,
    depths: Sequence[int] = tuple(range(1, 13)),
) -> dict:
    """Decision-tree depth sweep: the underfitting → overfitting arc.

    For each max_depth, fits a seeded DecisionTreeClassifier and records
    train and test accuracy AND error (error = 1 - accuracy) so the report
    can show train error monotonically decreasing while test error bottoms
    out and turns around — the textbook bias/variance trade-off.
    """
    X_train_list = [list(row) for row in X_train]
    X_test_list = [list(row) for row in X_test]
    y_train_list = list(y_train)
    y_test_list = list(y_test)
    table: list[dict] = []
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
        model.fit(X_train_list, y_train_list)
        train_accuracy = float(accuracy_score(y_train_list, model.predict(X_train_list)))
        test_accuracy = float(accuracy_score(y_test_list, model.predict(X_test_list)))
        table.append(
            {
                "depth": depth,
                "train_accuracy": round(train_accuracy, 4),
                "train_error": round(1.0 - train_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
                "test_error": round(1.0 - test_accuracy, 4),
            }
        )
    return {
        "model": "DecisionTreeClassifier(max_depth=d, random_state=42)",
        "table": table,
        "best_test_depth": max(table, key=lambda row: row["test_accuracy"])["depth"]
        if table else None,
    }


def regularization_demo(
    X_train: Sequence[Sequence[float]],
    y_train: Sequence,
    X_test: Sequence[Sequence[float]],
    y_test: Sequence,
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict:
    """LogisticRegression C sweep: stronger regularization, smaller train/test gap."""
    X_train_list = [list(row) for row in X_train]
    X_test_list = [list(row) for row in X_test]
    y_train_list = list(y_train)
    y_test_list = list(y_test)
    table: list[dict] = []
    for c_value in c_values:
        model = LogisticRegression(
            max_iter=2000, C=c_value, random_state=SEED
        )
        model.fit(X_train_list, y_train_list)
        train_accuracy = float(accuracy_score(y_train_list, model.predict(X_train_list)))
        test_accuracy = float(accuracy_score(y_test_list, model.predict(X_test_list)))
        table.append(
            {
                "C": c_value,
                "train_accuracy": round(train_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
                "gap": round(train_accuracy - test_accuracy, 4),
            }
        )
    return {
        "model": "LogisticRegression(max_iter=2000, C=c, random_state=42)",
        "table": table,
    }
