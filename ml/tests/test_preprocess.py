"""Tests for preprocess: dedupe, standardization, splitting, clean_rows."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ml/ -> gsms_ml
import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from gsms_ml.preprocess import (
    Dataset,
    Standardizer,
    chronological_split,
    clean_rows,
    load_raw_events,
    remove_near_duplicates,
)


def _row(headline: str, published: str = "2026-09-28T10:00:00+00:00",
         risk_band: str = "medium", region: str = "Europe", **overrides) -> dict:
    row = {
        "event_id": headline[:16],
        "headline": headline,
        "snippet": "",
        "source_domain": "example.com",
        "source_country": "GB",
        "published_utc": published,
        "region": region,
        "detected_countries": "GB",
        "sentiment_score": "-0.5",
        "corroboration_count": "2",
        "is_breaking": "0",
        "text_len": str(len(headline)),
        "word_count": str(len(headline.split())),
        "upper_ratio": "0.1",
        "velocity_per_hour": "0.5",
        "risk_score": "30",
        "risk_band": risk_band,
    }
    row.update(overrides)
    return row


class TestNearDuplicates(unittest.TestCase):
    def test_removes_near_duplicate_same_day(self):
        # same story reprinted with different punctuation: identical token sets
        rows = [
            _row("Russia launches missile attack on Kyiv"),
            _row("Russia launches missile attack on Kyiv!"),
            _row("Peace talks resume in Geneva"),
        ]
        kept = remove_near_duplicates(rows, threshold=0.9)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["headline"], rows[0]["headline"])

    def test_merely_similar_headlines_are_kept(self):
        # one inflected word apart: Jaccard 6/7 < 0.9, both stay
        rows = [
            _row("Russia launches missile attack on Kyiv"),
            _row("Russia launches missile attacks on Kyiv"),
        ]
        self.assertEqual(len(remove_near_duplicates(rows)), 2)

    def test_same_headline_different_day_kept(self):
        rows = [
            _row("Russia launches missile attack on Kyiv",
                 published="2026-09-28T10:00:00+00:00"),
            _row("Russia launches missile attack on Kyiv",
                 published="2026-09-29T10:00:00+00:00"),
        ]
        self.assertEqual(len(remove_near_duplicates(rows)), 2)


class TestStandardizer(unittest.TestCase):
    def test_fit_uses_train_statistics_only(self):
        train = [[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]]  # means (2, 14)
        test = [[12.0, 34.0]]  # far outside the train distribution
        standardizer = Standardizer().fit(train)
        self.assertAlmostEqual(standardizer.means[0], 2.0)
        self.assertAlmostEqual(standardizer.means[1], 14.0)
        transformed = standardizer.transform(test)
        # with train stats: (12 - 2) / std(0), not re-fitted on test
        std0 = (sum((v - 2.0) ** 2 for v in (0.0, 2.0, 4.0)) / 3.0) ** 0.5
        self.assertAlmostEqual(transformed[0][0], (12.0 - 2.0) / std0)
        # a different test set must not change the fitted parameters
        standardizer.transform([[100.0, 100.0]])
        self.assertAlmostEqual(standardizer.means[0], 2.0)

    def test_zero_variance_column_does_not_crash(self):
        standardizer = Standardizer().fit([[5.0], [5.0], [5.0]])
        self.assertEqual(standardizer.stds[0], 1.0)
        self.assertEqual(standardizer.transform([[5.0]])[0][0], 0.0)


class TestChronologicalSplit(unittest.TestCase):
    def _dataset(self, n: int = 20) -> Dataset:
        base = datetime.fromisoformat("2026-09-20T00:00:00+00:00")
        rows, timestamps = [], []
        for i in range(n):
            moment = base.replace(day=1 + i // 2, hour=i % 24)
            rows.append(_row(f"headline number {i} about region {i % 3}",
                             published=moment.isoformat()))
            timestamps.append(moment)
        dataset = clean_rows(rows)
        return dataset

    def test_ordering_and_no_overlap(self):
        dataset = self._dataset()
        train, validation, test = chronological_split(dataset)
        self.assertEqual(
            len(train.X) + len(validation.X) + len(test.X), len(dataset.X)
        )
        # strictly chronological: last train <= first validation <= last validation <= first test
        self.assertLessEqual(max(train.timestamps), min(validation.timestamps))
        self.assertLessEqual(max(validation.timestamps), min(test.timestamps))
        # no row object shared across splits (no overlap by identity of timestamps)
        train_keys = {ts.isoformat() for ts in train.timestamps}
        test_keys = {ts.isoformat() for ts in test.timestamps}
        self.assertFalse(train_keys & test_keys)
        # newest rows are in test
        self.assertEqual(
            max(ts for ts in dataset.timestamps), max(test.timestamps)
        )


class TestCleanRowsSyntheticCsv(unittest.TestCase):
    def test_missing_values_imputed_and_invalid_rows_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.csv"
            rows = [
                _row("Floods hit southern Asia hard"),
                _row("Floods hit southern Asia hard!"),  # same tokens: duplicate
                _row("Stocks rally after trade agreement"),
                # invalid row: broken timestamp
                _row("Mystery event", published="not-a-date"),
                # invalid row: unknown risk band
                _row("Another event", risk_band="catastrophic"),
            ]
            # punch a hole in one row's numeric feature -> median imputation
            rows[2]["corroboration_count"] = ""
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            loaded = load_raw_events(path)
            dataset = clean_rows(loaded)
            # 5 rows: 1 near-duplicate removed, 1 bad timestamp, 1 bad band -> 2 kept
            self.assertEqual(len(dataset.X), 2)
            self.assertEqual(dataset.removed_duplicate_count, 1)
            self.assertEqual(dataset.imputed_value_count, 1)
            self.assertEqual(len(dataset.feature_names), 7)
            # imputed value equals the median of the single valid value (2)
            self.assertEqual(dataset.X[1][1], 2.0)
            self.assertEqual(len(dataset.y_band), 2)
            self.assertEqual(len(dataset.headlines), 2)


if __name__ == "__main__":
    unittest.main()
