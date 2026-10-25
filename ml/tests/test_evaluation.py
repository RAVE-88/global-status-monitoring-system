"""Tests for evaluation: bootstrap CIs, CV fold counts, metric correctness."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ml/ -> gsms_ml
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from gsms_ml.evaluation import bootstrap_metric, cross_validate_accuracy
from gsms_ml.supervised import evaluate_model


class _FixedPredictor:
    """Stub estimator returning preset predictions/scores, for metric tests."""

    def __init__(self, predictions, probabilities=None):
        self._predictions = predictions
        self._probabilities = probabilities

    def fit(self, X, y):  # noqa: ARG002 - signature compatibility only
        self.classes_ = sorted(set(y))
        return self

    def predict(self, X):
        return np.asarray(self._predictions[: len(X)])

    def predict_proba(self, X):
        return np.asarray(self._probabilities[: len(X)])


class TestEvaluateModel(unittest.TestCase):
    def test_metrics_match_sklearn_on_fixed_predictions(self):
        y_train = [0, 0, 0, 0, 1, 1, 1, 1]
        y_test = [0, 0, 1, 1, 1]
        fixed_predictions = [0, 1, 1, 1, 0]
        X_train = [[0.0]] * len(y_train)
        X_test = [[0.0]] * len(y_test)
        result = evaluate_model(
            lambda: _FixedPredictor(fixed_predictions),
            X_train, y_train, X_test, y_test,
        )
        self.assertAlmostEqual(result["accuracy"],
                               accuracy_score(y_test, fixed_predictions))
        self.assertAlmostEqual(
            result["precision_macro"],
            precision_score(y_test, fixed_predictions, average="macro",
                            zero_division=0),
        )
        # confusion matrix over sorted union of labels [0, 1], row = truth
        self.assertEqual(result["labels"], ["0", "1"])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertEqual(result["n_test"], 5)

    def test_roc_auc_computed_from_predict_proba(self):
        y_train = [0, 0, 1, 1]
        y_test = [0, 1, 0, 1]
        scores = [0.1, 0.4, 0.35, 0.8]
        probabilities = [[1 - s, s] for s in scores]
        result = evaluate_model(
            lambda: _FixedPredictor([0, 1, 0, 1], probabilities),
            [[0.0]] * 4, y_train, [[0.0]] * 4, y_test,
        )
        expected = roc_auc_score(y_test, scores)
        self.assertAlmostEqual(result["roc_auc"], expected)

    def test_multiclass_has_no_roc_auc(self):
        y_train = [0, 1, 2, 0, 1, 2]
        y_test = [0, 1, 2]
        result = evaluate_model(
            lambda: _FixedPredictor([0, 1, 2]),
            [[0.0]] * 6, y_train, [[0.0]] * 3, y_test,
        )
        self.assertIsNone(result["roc_auc"])

    def test_perfect_separation_gives_perfect_metrics(self):
        # two far-apart gaussian blobs; every classifier should get 1.0
        X_train = [[-5.0, 0.0], [-4.0, 0.0], [-6.0, 0.0], [5.0, 0.0],
                   [4.0, 0.0], [6.0, 0.0]]
        y_train = [0, 0, 0, 1, 1, 1]
        X_test = [[-4.5, 0.0], [4.5, 0.0]]
        y_test = [0, 1]
        from gsms_ml.supervised import gaussian_nb_factory

        result = evaluate_model(gaussian_nb_factory, X_train, y_train,
                                X_test, y_test)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["f1_macro"], 1.0)
        flat = sum(sum(row) for row in result["confusion_matrix"])
        self.assertEqual(flat, len(y_test))


class TestBootstrap(unittest.TestCase):
    def test_ci_contains_point_estimate(self):
        y_true = [0, 1] * 50
        # predictor correct on 80 of the 100 samples
        y_pred = [t if i % 5 else 1 - t for i, t in enumerate(y_true)]
        point = accuracy_score(y_true, y_pred)
        result = bootstrap_metric(y_true, y_pred, accuracy_score, n_boot=200,
                                  seed=42)
        self.assertAlmostEqual(result["point"], point, places=4)
        self.assertLessEqual(result["lo"], result["point"])
        self.assertLessEqual(result["point"], result["hi"])
        # with 200 resamples the bootstrap mean tracks the point estimate
        self.assertAlmostEqual(result["mean"], point, delta=0.05)
        self.assertLessEqual(result["hi"] - result["lo"], 0.3)

    def test_deterministic_for_fixed_seed(self):
        y_true = [0, 1] * 20
        y_pred = [0, 1] * 20
        first = bootstrap_metric(y_true, y_pred, accuracy_score, n_boot=50, seed=7)
        second = bootstrap_metric(y_true, y_pred, accuracy_score, n_boot=50, seed=7)
        self.assertEqual(first, second)


class TestCrossValidation(unittest.TestCase):
    def test_fold_count_and_separation(self):
        # 30 well-separated samples per class, k=5
        X = [[0.0, 0.0]] * 30 + [[10.0, 10.0]] * 30
        y = [0] * 30 + [1] * 30
        from gsms_ml.supervised import gaussian_nb_factory

        result = cross_validate_accuracy(gaussian_nb_factory, X, y, k=5)
        self.assertEqual(len(result["fold_accuracies"]), 5)
        self.assertEqual(result["k"], 5)
        self.assertGreater(result["mean"], 0.99)
        self.assertAlmostEqual(result["std"], 0.0)

    def test_folds_shrink_when_a_class_is_too_rare(self):
        X = [[0.0]] * 40 + [[5.0]] * 3
        y = [0] * 40 + [1] * 3
        from gsms_ml.supervised import gaussian_nb_factory

        result = cross_validate_accuracy(gaussian_nb_factory, X, y, k=5)
        self.assertEqual(result["k"], 3)
        self.assertTrue(result["notes"])


if __name__ == "__main__":
    unittest.main()
