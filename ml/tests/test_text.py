"""Tests for the text layer: TF-IDF, cosine, LSA determinism, sentiment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ml/ -> gsms_ml
import unittest

from gsms_ml.text import (
    LsaEmbedder,
    TfidfVectorizer,
    cosine_similarity,
    preprocess_text,
    semantic_neighbours,
    text_features,
)
from gsms_ml.lexicon import headline_sentiment

TRAIN_DOCS = [
    "russia launches massive missile attack on ukraine capital kyiv",
    "ukraine reports russian missile strikes against kyiv districts",
    "peace talks begin in geneva between the two delegations",
    "ceasefire agreement reached after lengthy negotiations geneva",
    "strong earthquake hits tokyo japan buildings damaged",
    "earthquake aftershocks rattle japan coast triggering tsunami warning",
    "floods devastate villages across pakistan monsoon season",
    "wildfires spread across australia drought regions firefighters",
]


class TestPreprocessText(unittest.TestCase):
    def test_lowercases_and_drops_stopwords_and_short_tokens(self):
        tokens = preprocess_text("The US and UK vow to aid War-torn Regions")
        self.assertNotIn("the", tokens)
        self.assertNotIn("and", tokens)
        self.assertNotIn("to", tokens)
        self.assertIn("war-torn", tokens)
        self.assertIn("regions", tokens)
        self.assertIn("vow", tokens)
        # every token respects the [a-z][a-z-]* shape
        import re

        for token in tokens:
            self.assertRegex(token, r"^[a-z][a-z-]*$")

    def test_drops_tokens_shorter_than_three(self):
        self.assertEqual(preprocess_text("be as it may AI"), ["may"])


class TestTfidfVectorizer(unittest.TestCase):
    def test_cosine_of_identical_docs_is_one(self):
        vectorizer = TfidfVectorizer()
        rows = vectorizer.fit_transform(TRAIN_DOCS)
        self.assertAlmostEqual(cosine_similarity(rows[0], rows[0]), 1.0, places=9)
        self.assertAlmostEqual(cosine_similarity(rows[2], rows[2]), 1.0, places=9)
        # empty vs non-empty is 0
        self.assertEqual(cosine_similarity({}, rows[0]), 0.0)

    def test_fit_on_train_vocabulary_stable_across_transform(self):
        vectorizer = TfidfVectorizer(max_features=400, min_df=2)
        vectorizer.fit(TRAIN_DOCS)
        vocab_before = dict(vectorizer.vocabulary)
        idf_before = list(vectorizer.idf)
        # transform UNSEEN documents must not mutate the fitted state
        vectorizer.transform(["brand new unseen words about volcanoes everywhere",
                              "another unseen document about markets"])
        self.assertEqual(vectorizer.vocabulary, vocab_before)
        self.assertEqual(vectorizer.idf, idf_before)
        # unseen-only terms never enter the vocabulary
        self.assertNotIn("volcanoes", vectorizer.vocabulary)

    def test_deterministic_and_bounded(self):
        first = TfidfVectorizer(max_features=10)
        second = TfidfVectorizer(max_features=10)
        rows_a = first.fit_transform(TRAIN_DOCS)
        rows_b = second.fit_transform(TRAIN_DOCS)
        self.assertEqual(first.vocabulary, second.vocabulary)
        self.assertEqual(first.idf, second.idf)
        self.assertEqual(rows_a, rows_b)
        self.assertLessEqual(len(first.vocabulary), 10)


class TestLsaEmbedder(unittest.TestCase):
    def test_deterministic_embeddings(self):
        first = LsaEmbedder(rank=4, iterations=5).fit_transform(TRAIN_DOCS)
        second = LsaEmbedder(rank=4, iterations=5).fit_transform(TRAIN_DOCS)
        self.assertEqual(first, second)

    def test_embedding_dimension_is_rank(self):
        embedder = LsaEmbedder(rank=4, iterations=5).fit(TRAIN_DOCS)
        vectors = embedder.transform(TRAIN_DOCS)
        self.assertEqual(len(vectors), len(TRAIN_DOCS))
        for vector in vectors:
            self.assertEqual(len(vector), 4)

    def test_related_docs_score_higher_than_unrelated(self):
        embedder = LsaEmbedder(rank=4, iterations=10).fit(TRAIN_DOCS)
        vectors = embedder.transform(TRAIN_DOCS)

        from gsms_ml.text import _cosine_dense

        same_topic = _cosine_dense(vectors[0], vectors[1])   # two strike stories
        cross_topic = _cosine_dense(vectors[0], vectors[4])  # strike vs earthquake
        self.assertGreaterEqual(same_topic, cross_topic)

    def test_semantic_neighbours_ranks_self_first(self):
        embedder = LsaEmbedder(rank=4, iterations=10).fit(TRAIN_DOCS)
        query = TRAIN_DOCS[2]
        neighbours = semantic_neighbours(query, TRAIN_DOCS, embedder, k=3)
        self.assertEqual(len(neighbours), 3)
        self.assertEqual(neighbours[0][0], 2)
        self.assertAlmostEqual(neighbours[0][1], 1.0, places=6)


class TestSentimentAndFeatures(unittest.TestCase):
    def test_negative_headline_negative_score(self):
        self.assertLess(
            headline_sentiment("Dozens killed in devastating missile attack"), 0.0
        )

    def test_positive_headline_positive_score(self):
        self.assertGreater(
            headline_sentiment("Historic peace agreement brings hope of recovery"),
            0.0,
        )

    def test_negation_flips_polarity(self):
        # minimal pair: the single scored word follows the negator
        plain = headline_sentiment("the economy faces crisis")
        negated = headline_sentiment("the economy faces no crisis")
        self.assertLess(plain, 0.0)
        self.assertGreater(negated, 0.0)

    def test_neutral_text_scores_zero(self):
        self.assertEqual(headline_sentiment("the delegation arrived by train"), 0.0)

    def test_text_features_combines_layers(self):
        features = text_features("Russian missile attack near Kyiv alarms NATO")
        self.assertLess(features["sentiment"], 0.0)
        self.assertIn("RU", features["countries"])
        self.assertIn("UA", features["countries"])
        self.assertIn("missile", features["tokens"])
        self.assertIn("attack", features["tokens"])


if __name__ == "__main__":
    unittest.main()
