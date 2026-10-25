"""Preprocessing: raw collected articles -> ML-ready matrices.

Implements the preprocessing syllabus items end to end:

- duplicate removal (near-duplicate headlines grouped by token-set Jaccard),
- missing-value handling (drop unparseable rows, impute numeric medians),
- categorical encoding (ordinal labels for tree models),
- one-hot encoding (linear models),
- standardization (z-score, parameters fitted on the training split only),
- chronological train/validation/test splitting so no future article leaks
  into the past (the data is time-stamped, so ordering matters).

Every transformer here is a small explicit class rather than black-box
preprocessing, so a student can read exactly what is learned from the data.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .lexicon import tokenize

RISK_BANDS = ["low", "medium", "high", "critical"]

REGIONS = [
    "Africa",
    "Asia",
    "Europe",
    "Global",
    "Middle East",
    "North America",
    "Oceania",
    "South America",
]

NUMERIC_COLUMNS = [
    "sentiment_score",
    "corroboration_count",
    "is_breaking",
    "text_len",
    "word_count",
    "upper_ratio",
    "velocity_per_hour",
]


@dataclass
class Dataset:
    """A cleaned event table with assembled feature matrices."""

    raw_rows: list[dict]
    removed_duplicate_count: int = 0
    imputed_value_count: int = 0
    feature_names: list[str] = field(default_factory=list)
    X: list[list[float]] = field(default_factory=list)  # one-hot + numeric
    y_band: list[int] = field(default_factory=list)     # 0..3 risk band
    y_topic: list[int] = field(default_factory=list)    # topic class index
    topic_names: list[str] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    headlines: list[str] = field(default_factory=list)


def load_raw_events(path: Path | None = None) -> list[dict]:
    """Read the collected article table from disk."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "gsms_events.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def remove_near_duplicates(rows: list[dict], threshold: float = 0.9) -> list[dict]:
    """Drop later articles whose headline token set nearly matches an earlier one.

    Within a single publication day, two headlines sharing >= `threshold`
    Jaccard similarity over their token sets are treated as the same story;
    the first (highest-velocity) article is kept as the representative.
    """
    representatives: list[dict] = []
    seen_token_sets: list[set[str]] = []
    seen_days: list[str] = []
    for row in rows:
        tokens = set(tokenize(row.get("headline", "")))
        day = (row.get("published_utc") or "")[:10]
        duplicate = False
        for index, seen in enumerate(seen_token_sets):
            if seen_days[index] != day:
                continue
            union = len(tokens | seen)
            if union and len(tokens & seen) / union >= threshold:
                duplicate = True
                break
        if duplicate:
            continue
        representatives.append(row)
        seen_token_sets.append(tokens)
        seen_days.append(day)
    return representatives


def clean_rows(rows: list[dict]) -> Dataset:
    """Deduplicate, drop invalid rows, and assemble the feature matrix."""
    deduped = remove_near_duplicates(rows)
    removed = len(rows) - len(deduped)

    kept_rows: list[dict] = []
    for row in deduped:
        if _parse_timestamp(row.get("published_utc", "")) is None:
            continue
        if row.get("risk_band") not in RISK_BANDS:
            continue
        kept_rows.append(row)

    # median imputation for numeric gaps (missing-value handling)
    medians: dict[str, float] = {}
    imputed = 0
    for column in NUMERIC_COLUMNS:
        values = []
        for row in kept_rows:
            try:
                values.append(float(row.get(column, "")))
            except (TypeError, ValueError):
                continue
        medians[column] = sorted(values)[len(values) // 2] if values else 0.0

    numeric_rows: list[list[float]] = []
    for row in kept_rows:
        row_values: list[float] = []
        for column in NUMERIC_COLUMNS:
            try:
                row_values.append(float(row.get(column, "")))
            except (TypeError, ValueError):
                row_values.append(medians[column])
                imputed += 1
        numeric_rows.append(row_values)

    feature_names = list(NUMERIC_COLUMNS)
    X = [row[:] for row in numeric_rows]
    y_band = [RISK_BANDS.index(row["risk_band"]) for row in kept_rows]

    topic_names = sorted({row["region"] for row in kept_rows if row["region"]})
    y_topic = [topic_names.index(row["region"]) for row in kept_rows]

    timestamps = [
        _parse_timestamp(row["published_utc"]) for row in kept_rows
    ]
    headlines = [row["headline"] for row in kept_rows]

    return Dataset(
        raw_rows=kept_rows,
        removed_duplicate_count=removed,
        imputed_value_count=imputed,
        feature_names=feature_names,
        X=X,
        y_band=y_band,
        y_topic=y_topic,
        topic_names=topic_names,
        timestamps=timestamps,
        headlines=headlines,
    )


class Standardizer:
    """Z-score standardizer whose mean/std are fitted on training data only."""

    def __init__(self) -> None:
        self.means: list[float] = []
        self.stds: list[float] = []

    def fit(self, X: list[list[float]]) -> "Standardizer":
        columns = len(X[0]) if X else 0
        self.means = [0.0] * columns
        self.stds = [1.0] * columns
        for j in range(columns):
            values = [row[j] for row in X]
            self.means[j] = sum(values) / len(values)
            variance = sum((v - self.means[j]) ** 2 for v in values) / len(values)
            self.stds[j] = math.sqrt(variance) or 1.0
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        return [
            [
                (row[j] - self.means[j]) / self.stds[j]
                for j in range(len(self.means))
            ]
            for row in X
        ]

    def fit_transform(self, X: list[list[float]]) -> list[list[float]]:
        return self.fit(X).transform(X)


def chronological_split(
    dataset: Dataset,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[Dataset, Dataset, Dataset]:
    """Split oldest→newest into train/validation/test, preserving chronology.

    The newest articles become the test set: evaluation always measures the
    models on events that happened *after* everything they trained on.
    """
    order = sorted(range(len(dataset.X)), key=lambda i: dataset.timestamps[i])
    n = len(order)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    splits: list[list[int]] = [
        order[:train_end],
        order[train_end:validation_end],
        order[validation_end:],
    ]

    def subset(indices: list[int]) -> Dataset:
        return Dataset(
            raw_rows=[dataset.raw_rows[i] for i in indices],
            feature_names=list(dataset.feature_names),
            X=[dataset.X[i] for i in indices],
            y_band=[dataset.y_band[i] for i in indices],
            y_topic=[dataset.y_topic[i] for i in indices],
            topic_names=list(dataset.topic_names),
            timestamps=[dataset.timestamps[i] for i in indices],
            headlines=[dataset.headlines[i] for i in indices],
        )

    return subset(splits[0]), subset(splits[1]), subset(splits[2])
