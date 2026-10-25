"""Supervised classification wrappers: thin, documented sklearn adapters.

Each classifier the syllabus requires is exposed as a *factory* — a zero-
argument callable returning a fresh, seeded estimator — so that every fit
starts from identical hyperparameters and every experiment (holdout, k-fold
CV, bootstrap) can rebuild the model independently. All random_state values
are pinned to 42.

``evaluate_model`` is the single entry point used by every experiment: it
fits on the training split only, predicts the held-out split once, and
reports the full metric suite (accuracy, macro precision/recall/F1, the
confusion matrix over the union of train+test labels, and — for binary
targets — ROC-AUC from ``predict_proba`` or ``decision_function``).

Labels are derived heuristics (keyword-classifier risk bands), never ground
truth; the metrics measure agreement with that labeling function.
"""

from __future__ import annotations

from typing import Callable, Sequence

from sklearn.ensemble import RandomForestClassifier  # noqa: F401  (re-export)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier  # noqa: F401  (re-export)
from sklearn.linear_model import LogisticRegression

SEED = 42

ModelFactory = Callable[[], object]


def gaussian_nb_factory() -> GaussianNB:
    """Gaussian Naive Bayes (no hyperparameters to tune; deterministic)."""
    return GaussianNB()


def knn_factory(k: int = 5) -> KNeighborsClassifier:
    """k-Nearest Neighbours, k=5, distance-weighted voting."""
    return KNeighborsClassifier(n_neighbors=k, weights="distance")


def logistic_factory(max_iter: int = 2000, c: float = 1.0) -> LogisticRegression:
    """Logistic Regression (multinomial, lbfgs), capped iterations, seeded."""
    return LogisticRegression(max_iter=max_iter, C=c, random_state=SEED)


def linear_svc_factory() -> LinearSVC:
    """Linear Support Vector Machine (squared-hinge one-vs-rest)."""
    return LinearSVC(random_state=SEED, max_iter=5000, dual="auto")


def rbf_svc_factory() -> SVC:
    """RBF-kernel Support Vector Machine (decision_function for ROC-AUC)."""
    return SVC(kernel="rbf", random_state=SEED)


#: display name -> factory, used by run_all to iterate every classifier
MODEL_FACTORIES: dict[str, ModelFactory] = {
    "GaussianNB": gaussian_nb_factory,
    "kNN(k=5,distance)": knn_factory,
    "LogisticRegression": logistic_factory,
    "LinearSVC": linear_svc_factory,
    "SVC(rbf)": rbf_svc_factory,
}


def _binary_roc_auc(estimator, X_test, y_test) -> float | None:
    """ROC-AUC for binary targets; None when unavailable or degenerate."""
    classes_raw = getattr(estimator, "classes_", None)
    classes = [] if classes_raw is None else list(classes_raw)
    if len(classes) != 2:
        return None
    try:
        unique_test = set(y_test)
        if len(unique_test) < 2:
            return None
        if hasattr(estimator, "predict_proba"):
            column = classes.index(classes[1])
            scores = estimator.predict_proba(X_test)[:, column]
        elif hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(X_test)
        else:
            return None
        return float(roc_auc_score(y_test, scores))
    except (ValueError, TypeError):
        return None


def evaluate_model(
    model_factory: ModelFactory,
    X_train: Sequence[Sequence[float]],
    y_train: Sequence,
    X_test: Sequence[Sequence[float]],
    y_test: Sequence,
) -> dict:
    """Fit a fresh model on the train split and score it on the test split.

    Returns accuracy, macro precision/recall/F1 (zero_division=0 so rare
    classes never crash the suite), the confusion matrix over the sorted union
    of train+test labels, and ROC-AUC when the target is binary (via
    predict_proba, falling back to decision_function). Never mutates shared
    state: a new estimator is built per call.
    """
    X_train_list = [list(row) for row in X_train]
    X_test_list = [list(row) for row in X_test]
    y_train_list = list(y_train)
    y_test_list = list(y_test)

    estimator = model_factory()
    estimator.fit(X_train_list, y_train_list)
    predictions = estimator.predict(X_test_list)

    labels = sorted(set(y_train_list) | set(y_test_list), key=str)
    matrix = confusion_matrix(y_test_list, predictions, labels=labels)

    return {
        "model": type(estimator).__name__,
        "accuracy": float(accuracy_score(y_test_list, predictions)),
        "precision_macro": float(
            precision_score(y_test_list, predictions, average="macro",
                            zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test_list, predictions, average="macro",
                         zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test_list, predictions, average="macro", zero_division=0)
        ),
        "roc_auc": _binary_roc_auc(estimator, X_test_list, y_test_list),
        "confusion_matrix": [[int(cell) for cell in row] for row in matrix],
        "labels": [str(label) for label in labels],
        "n_train": len(y_train_list),
        "n_test": len(y_test_list),
    }
