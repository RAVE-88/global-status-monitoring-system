"""Ensemble methods: Random Forest, AdaBoost, Bagging + feature importance.

Three ensemble families over the same standardized features the base
classifiers use, reusing ``supervised.evaluate_model`` for the metric suite:

- Random Forest: 100 bagged decision trees with random feature subsampling
  (random_state=42); its ``feature_importances_`` (mean decrease in impurity)
  is mapped back to feature names — the lab's interpretability story.
- AdaBoost: 100 rounds of 50/50 re-weighting over depth-2 decision stumps,
  learning_rate 0.5. ``algorithm="SAMME"`` is passed explicitly because
  scikit-learn 1.5 deprecates the SAMME.R default (removed in 1.6).
- Bagging: 50 bootstrap-resampled decision stumps (max_depth=1). Stumps are
  the classic unstable base learner that bagging stabilizes; a
  LogisticRegression base is supported via ``base_factory`` but stumps are
  the default experiment.

A fresh ensemble is fitted per call (every factory returns a new estimator),
so results are reproducible and independent of call order.
"""

from __future__ import annotations

from typing import Callable, Sequence

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from .supervised import SEED, evaluate_model

EnsembleFactory = Callable[[], object]


def random_forest_factory() -> RandomForestClassifier:
    """Random Forest: 100 trees, seeded."""
    return RandomForestClassifier(n_estimators=100, random_state=SEED)


def ada_boost_factory() -> AdaBoostClassifier:
    """AdaBoost over depth-2 trees; SAMME (sklearn 1.5 deprecates SAMME.R)."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=100,
        learning_rate=0.5,
        algorithm="SAMME",
        random_state=SEED,
    )


def bagging_factory(base_factory: Callable[[], object] | None = None) -> BaggingClassifier:
    """Bagging over decision stumps (or any provided base estimator)."""
    if base_factory is None:
        base_factory = lambda: DecisionTreeClassifier(max_depth=1)  # noqa: E731
    return BaggingClassifier(
        estimator=base_factory(),
        n_estimators=50,
        random_state=SEED,
    )


#: display name -> factory, used by run_all
ENSEMBLE_FACTORIES: dict[str, EnsembleFactory] = {
    "RandomForest": random_forest_factory,
    "AdaBoost": ada_boost_factory,
    "Bagging(stumps)": bagging_factory,
}


def _feature_importances(
    model_factory: EnsembleFactory,
    X_train: Sequence[Sequence[float]],
    y_train: Sequence,
    feature_names: Sequence[str],
) -> list[dict] | None:
    """Refit a fresh ensemble and map ``feature_importances_`` to names.

    The attribute is checked AFTER fitting: ``feature_importances_`` is a
    property that raises NotFittedError on a fresh estimator, so a pre-fit
    ``hasattr`` check would wrongly report every ensemble as unsupported.
    Returns None for ensembles without the attribute (Bagging over stumps
    does not expose a mean impurity importance in sklearn 1.5).
    """
    estimator = model_factory()
    X_train_list = [list(row) for row in X_train]
    estimator.fit(X_train_list, list(y_train))
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None
    ranked = sorted(
        zip(feature_names, importances),
        key=lambda pair: -float(pair[1]),
    )
    return [
        {"feature": name, "importance": round(float(importance), 4)}
        for name, importance in ranked
    ]


def evaluate_ensemble(
    model_factory: EnsembleFactory,
    X_train: Sequence[Sequence[float]],
    y_train: Sequence,
    X_test: Sequence[Sequence[float]],
    y_test: Sequence,
    feature_names: Sequence[str] = (),
) -> dict:
    """Metrics via ``supervised.evaluate_model`` + feature importances."""
    result = evaluate_model(model_factory, X_train, y_train, X_test, y_test)
    importances = _feature_importances(model_factory, X_train, y_train, feature_names)
    if importances is not None:
        result["feature_importances"] = importances
    return result
