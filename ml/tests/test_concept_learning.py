"""Tests for concept_learning: Find-S convergence and Candidate Elimination."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ml/ -> gsms_ml
import unittest

from gsms_ml.concept_learning import candidate_elimination, covers, find_s


# Mitchell's EnjoySport training set (Table 2.2). Attributes in order:
# Sky, AirTemp, Humidity, Wind, Water, Forecast.
ENJOY_SPORT = [
    {"Sky": "Sunny", "AirTemp": "Warm", "Humidity": "Normal", "Wind": "Strong",
     "Water": "Warm", "Forecast": "Same", "EnjoySport": "Yes"},
    {"Sky": "Sunny", "AirTemp": "Warm", "Humidity": "High", "Wind": "Strong",
     "Water": "Warm", "Forecast": "Same", "EnjoySport": "Yes"},
    {"Sky": "Rainy", "AirTemp": "Cold", "Humidity": "High", "Wind": "Strong",
     "Water": "Warm", "Forecast": "Change", "EnjoySport": "No"},
    {"Sky": "Sunny", "AirTemp": "Warm", "Humidity": "High", "Wind": "Strong",
     "Water": "Cool", "Forecast": "Change", "EnjoySport": "Yes"},
]

ENJOY_SPORT_EXPECTED = {
    "Sky": "Sunny", "AirTemp": "Warm", "Humidity": "?", "Wind": "Strong",
    "Water": "?", "Forecast": "?",
}


class TestFindS(unittest.TestCase):
    def test_converges_on_mitchell_enjoysport(self):
        self.assertEqual(find_s(ENJOY_SPORT, "EnjoySport", "Yes"),
                         ENJOY_SPORT_EXPECTED)

    def test_trace_snapshots_per_example(self):
        from gsms_ml.concept_learning import find_s_trace

        trace = find_s_trace(ENJOY_SPORT, "EnjoySport", "Yes")
        self.assertEqual(len(trace), len(ENJOY_SPORT))
        # first positive initializes the hypothesis to the example itself
        self.assertEqual(trace[0]["hypothesis"]["Sky"], "Sunny")
        # negative example must leave the hypothesis unchanged
        self.assertEqual(trace[2]["hypothesis"], trace[1]["hypothesis"])
        self.assertFalse(trace[2]["is_positive"])
        # final snapshot equals the converged hypothesis
        self.assertEqual(trace[-1]["hypothesis"], ENJOY_SPORT_EXPECTED)

    def test_no_positive_examples_gives_empty_hypothesis(self):
        self.assertEqual(find_s(ENJOY_SPORT, "EnjoySport", "Maybe"), {})

    def test_semantics_of_covers(self):
        hypothesis = {"Sky": "Sunny", "AirTemp": "?"}
        sunny_example = {"Sky": "Sunny", "AirTemp": "Cold"}
        rainy_example = {"Sky": "Rainy", "AirTemp": "Cold"}
        self.assertTrue(covers(hypothesis, sunny_example, ["Sky", "AirTemp"]))
        self.assertFalse(covers(hypothesis, rainy_example, ["Sky", "AirTemp"]))


class TestCandidateElimination(unittest.TestCase):
    def test_boundaries_on_linearly_separable_toy(self):
        # target concept: a == "x" (attribute b is irrelevant noise).
        # With one positive block, S and G converge to the unique consistent
        # hypothesis {a: x, b: ?} — Mitchell's termination condition.
        examples = [
            {"a": "x", "b": "p1", "label": "pos"},
            {"a": "x", "b": "p2", "label": "pos"},
            {"a": "y", "b": "p3", "label": "neg"},
            {"a": "z", "b": "p4", "label": "neg"},
        ]
        result = candidate_elimination(examples, "label", "pos")
        self.assertEqual(result["specific"], [{"a": "x", "b": "?"}])
        self.assertEqual(result["general"], [{"a": "x", "b": "?"}])
        self.assertEqual(result["version_space_size"], 1)
        self.assertEqual(result["skipped_examples"], 0)
        self.assertEqual(result["notes"], [])

    def test_every_boundary_consistent_with_training_data(self):
        examples = [
            {"a": "x", "b": "warm", "label": "1"},
            {"a": "x", "b": "cold", "label": "1"},
            {"a": "y", "b": "warm", "label": "0"},
            {"a": "z", "b": "cold", "label": "0"},
            {"a": "x", "b": "cool", "label": "1"},
            {"a": "w", "b": "warm", "label": "0"},
        ]
        result = candidate_elimination(examples, "label", "1")
        attributes = ["a", "b"]
        for s in result["specific"]:
            for example in examples:
                covered = covers(s, example, attributes)
                self.assertEqual(
                    covered, example["label"] == "1",
                    f"S member {s} misclassifies {example}",
                )
        for g in result["general"]:
            for example in examples:
                covered = covers(g, example, attributes)
                if example["label"] == "0":
                    self.assertFalse(covered, f"G member {g} covers negative {example}")

    def test_noise_skipped_with_notes_not_crash(self):
        consistent = [
            {"a": "x", "b": "p1", "label": "pos"},
            {"a": "x", "b": "p2", "label": "pos"},
            {"a": "y", "b": "p3", "label": "neg"},
        ]
        # noisy positive: contradicts every general boundary's values
        noisy = {"a": "y", "b": "p9", "label": "pos"}
        result = candidate_elimination(consistent + [noisy], "label", "pos")
        self.assertEqual(result["specific"], [{"a": "x", "b": "?"}])
        self.assertEqual(result["skipped_examples"], 1)
        self.assertTrue(result["notes"])
        # history records the skip
        self.assertTrue(result["history"][-1]["skipped"])

    def test_history_tracks_boundaries_per_example(self):
        examples = [
            {"a": "x", "b": "p1", "label": "pos"},
            {"a": "y", "b": "p2", "label": "neg"},
        ]
        result = candidate_elimination(examples, "label", "pos")
        self.assertEqual(len(result["history"]), 2)
        # after the first (positive) example, S is the example itself
        self.assertEqual(result["history"][0]["specific"],
                         [{"a": "x", "b": "p1"}])


if __name__ == "__main__":
    unittest.main()
