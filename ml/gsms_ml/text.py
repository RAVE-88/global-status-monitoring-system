"""Text layer: tokenization, TF-IDF, latent-semantic embeddings, cosine similarity.

The syllabus NLP layer asks for TF-IDF, embeddings, semantic similarity and
event clustering. This module implements a fully offline, deterministic
pipeline:

- unigram tokenization with stopword removal (``preprocess_text``),
- TF-IDF vectorization with idf fitted on the training corpus only
  (``TfidfVectorizer``), producing sparse rows that cosine similarity can
  consume cheaply,
- latent-semantic analysis embeddings: the sparse TF-IDF matrix is projected
  onto its top-k right singular directions, approximated by subspace
  (orthogonal) iteration with modified Gram-Schmidt re-orthonormalization —
  the deterministic, dependency-light substitute for transformer embeddings,
- cosine similarity and a nearest-neighbour semantic retrieval helper
  (``semantic_neighbours``),
- one-call feature assembly (``text_features``) combining the sentiment
  lexicon, gazetteer NER and tokens.

Determinism: no RNG anywhere. The subspace iteration starts from the fixed
matrix V[i][j] = sin(0.735*i + 2.318*j), runs a fixed number of iterations,
and every operation is plain float arithmetic, so repeated fits on the same
corpus give bit-identical embeddings.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .gazetteer import detect_countries
from .lexicon import headline_sentiment

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "had", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "will", "with", "after", "over", "into", "amid",
    "amidst", "about", "across", "ahead", "before", "during", "under",
    "between", "against", "without", "within", "this", "these", "those",
    "their", "his", "her", "our", "your", "they", "them", "he", "she", "we",
    "you", "i", "who", "whom", "which", "what", "when", "where", "why",
    "how", "all", "any", "both", "each", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "just", "should", "now", "says", "said", "say", "new",
    "report", "reports",
}

_TOKEN_RE = re.compile(r"[a-z][a-z-]*")


def preprocess_text(text: str) -> list[str]:
    """Lowercase ``[a-z][a-z-]*`` tokens with stopwords and short words dropped."""
    return [
        raw
        for raw in _TOKEN_RE.findall(text.lower())
        if len(raw) >= 3 and raw not in STOPWORDS
    ]


class TfidfVectorizer:
    """TF-IDF vectorizer with the vocabulary and idf fitted on the corpus.

    ``fit`` must see only the training documents; later ``transform`` calls on
    unseen documents reuse the frozen vocabulary and idf (fit-on-train
    discipline — no leakage). Rows are sparse dicts ``{column: tfidf}`` which
    makes cosine similarity and the LSA multiplies cheap.
    """

    def __init__(self, max_features: int = 400, min_df: int = 2) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []

    def fit(self, documents: list[str]) -> "TfidfVectorizer":
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(preprocess_text(document)))
        # Deterministic selection: keep terms with df >= min_df, rank by
        # (descending document frequency, then alphabetical) and cut to the
        # cap; the final vocabulary is ordered alphabetically.
        kept = [
            (term, count)
            for term, count in sorted(document_frequency.items())
            if count >= self.min_df
        ]
        kept.sort(key=lambda item: (-item[1], item[0]))
        kept_terms = sorted(term for term, _ in kept[: self.max_features])
        self.vocabulary = {term: index for index, term in enumerate(kept_terms)}
        total_documents = max(len(documents), 1)
        self.idf = [
            math.log((1 + total_documents) / (1 + document_frequency[term])) + 1.0
            for term in kept_terms
        ]
        return self

    def transform(self, documents: list[str]) -> list[dict[int, float]]:
        """Sparse TF-IDF rows; terms outside the fitted vocabulary are ignored."""
        vectors: list[dict[int, float]] = []
        for document in documents:
            counts = Counter(preprocess_text(document))
            total = sum(counts.values())
            row: dict[int, float] = {}
            for term, count in counts.items():
                index = self.vocabulary.get(term)
                if index is None or not total:
                    continue
                row[index] = (count / total) * self.idf[index]
            vectors.append(row)
        return vectors

    def fit_transform(self, documents: list[str]) -> list[dict[int, float]]:
        return self.fit(documents).transform(documents)


def cosine_similarity(a: dict[int, float], b: dict[int, float]) -> float:
    """Cosine similarity between two sparse vectors (0.0 if either is empty)."""
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_dense(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _orthonormalize_columns(matrix: list[list[float]]) -> list[list[float]]:
    """Modified Gram-Schmidt over the columns of a row-major matrix.

    Column j is repeatedly projected against each already-orthonormalized
    earlier column, then normalized. Near-zero columns (rank deficiency) are
    zeroed so the result stays a valid orthonormal basis of the span.
    """
    rows = len(matrix)
    if rows == 0:
        return []
    cols = len(matrix[0])
    # work in column-major form for cache-friendly list operations
    columns = [[matrix[r][c] for r in range(rows)] for c in range(cols)]
    for j in range(cols):
        column = columns[j]
        for i in range(j):
            earlier = columns[i]
            dot = sum(a * b for a, b in zip(column, earlier))
            if dot:
                column = [a - dot * b for a, b in zip(column, earlier)]
        norm = math.sqrt(sum(a * a for a in column))
        if norm < 1e-12:
            column = [0.0] * rows
        else:
            inv = 1.0 / norm
            column = [a * inv for a in column]
        columns[j] = column
    return [[columns[c][r] for c in range(cols)] for r in range(rows)]


class LsaEmbedder:
    """Latent-semantic document embeddings via subspace iteration on TF-IDF.

    Given the sparse TF-IDF matrix X (n_docs x n_terms), the top-k right
    singular directions V (n_terms x k) are approximated by orthogonal
    iteration:

        repeat ``iterations`` times:
            Z = X @ V            (n_docs x k)   document-space projections
            Z = mgs_orthonormalize(Z)
            W = X^T @ Z          (n_terms x k)  back to term space
            V = mgs_orthonormalize(W)

    which is orthogonal iteration on X^T X and converges to the dominant
    k-dimensional right singular subspace of X. Document embeddings are the
    rows of X @ V_final (dense, k-dimensional); cosine similarity in this
    space reflects term co-occurrence structure ("semantic" similarity).

    Determinism: no RNG. The initial subspace is the fixed table
    V[i][j] = sin(0.735*(i+1)*(j+1) + 2.318*(j+1)) — the textbook-style
    additive variant sin(0.735*i + 2.318*j) was rejected deliberately: by
    angle addition sin(a_i + b_j) = sin(a_i)cos(b_j) + cos(a_i)sin(b_j), so
    that table has EXACTLY rank 2 and subspace iteration could never leave
    a 2-dimensional start (on the real corpus it captured 14% of the top-32
    singular energy; this init captures 99.9%). Fixed iteration count,
    rank min(32, n_terms - 1), pure float arithmetic throughout — repeated
    fits on the same corpus give bit-identical embeddings.
    """

    def __init__(self, rank: int = 32, iterations: int = 20,
                 max_features: int = 400, min_df: int = 2) -> None:
        self.rank = rank
        self.iterations = iterations
        self.max_features = max_features
        self.min_df = min_df
        self.vectorizer: TfidfVectorizer | None = None
        self.projection: list[list[float]] = []  # n_terms x k, row-major
        self.term_index: dict[str, int] = {}

    def fit(self, documents: list[str]) -> "LsaEmbedder":
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, min_df=self.min_df
        )
        sparse_rows = self.vectorizer.fit_transform(documents)
        self.term_index = dict(self.vectorizer.vocabulary)
        n_terms = len(self.term_index)
        k = min(self.rank, n_terms - 1)
        if k < 1 or not sparse_rows:
            # Degenerate corpus (fewer than 2 distinct kept terms): emit
            # fixed-dimension zero embeddings so callers still get vectors.
            self.projection = []
            self.rank = max(k, 1)
            return self

        # deterministic full-rank initial subspace, n_terms x k (see the
        # class docstring for why the additive sin table is not usable)
        v = [
            [math.sin(0.735 * (i + 1) * (j + 1) + 2.318 * (j + 1)) for j in range(k)]
            for i in range(n_terms)
        ]
        for _ in range(self.iterations):
            z = self._multiply(sparse_rows, v)          # X @ V
            z = _orthonormalize_columns(z)
            w = self._transpose_multiply(sparse_rows, z)  # X^T @ Z
            v = _orthonormalize_columns(w)
        self.projection = v
        return self

    @property
    def embedding_dim(self) -> int:
        if self.projection:
            return len(self.projection[0])
        return self.rank if self.rank >= 1 else 1

    def _multiply(
        self, sparse_rows: list[dict[int, float]], matrix: list[list[float]]
    ) -> list[list[float]]:
        """X @ matrix for sparse X rows and a dense n_terms x k matrix."""
        k = len(matrix[0]) if matrix else 0
        out: list[list[float]] = []
        for row in sparse_rows:
            acc = [0.0] * k
            for key, value in row.items():
                mrow = matrix[key]
                acc = [a + value * b for a, b in zip(acc, mrow)]
            out.append(acc)
        return out

    def _transpose_multiply(
        self, sparse_rows: list[dict[int, float]], matrix: list[list[float]]
    ) -> list[list[float]]:
        """X^T @ matrix for sparse X rows and a dense n_docs x k matrix."""
        n_terms = len(self.term_index)
        k = len(matrix[0]) if matrix else 0
        out = [[0.0] * k for _ in range(n_terms)]
        for doc_index, row in enumerate(sparse_rows):
            mrow = matrix[doc_index]
            for key, value in row.items():
                orow = out[key]
                out[key] = [o + value * m for o, m in zip(orow, mrow)]
        return out

    def transform(self, documents: list[str]) -> list[list[float]]:
        """Embed documents as rows of X @ V using the fitted projection."""
        if self.vectorizer is None:
            raise RuntimeError("LsaEmbedder.fit must be called before transform")
        if not self.projection:
            return [[0.0] * self.embedding_dim for _ in documents]
        sparse_rows = self.vectorizer.transform(documents)
        return self._multiply(sparse_rows, self.projection)

    def fit_transform(self, documents: list[str]) -> list[list[float]]:
        self.fit(documents)
        return self.transform(documents)


def semantic_neighbours(
    query: str, documents: list[str], embedder: LsaEmbedder, k: int = 5
) -> list[tuple[int, float]]:
    """Indices + cosine scores of the k documents semantically closest to query.

    Embeds the query and every candidate document in the fitted LSA space and
    ranks by cosine similarity (descending). Ties break by index so the result
    is deterministic.
    """
    query_vec = embedder.transform([query])[0]
    doc_vecs = embedder.transform(documents)
    scored = [
        (index, _cosine_dense(query_vec, vec))
        for index, vec in enumerate(doc_vecs)
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:k]


def text_features(text: str) -> dict:
    """One-call interpretable text features: sentiment, countries, tokens.

    Combines the committed sentiment lexicon (``lexicon.headline_sentiment``),
    the gazetteer country recognizer (``gazetteer.detect_countries``) and the
    preprocessed token list. All three are transparent heuristics.
    """
    return {
        "sentiment": headline_sentiment(text),
        "countries": detect_countries(text),
        "tokens": preprocess_text(text),
    }
