"""Unsupervised clustering: K-Means, hierarchical, DBSCAN, EM/GMM + PCA.

Runs the four clustering families over the standardized numeric feature
matrix, plus a fifth "semantic" experiment: K-Means over LSA text embeddings
of the headlines (``text.LsaEmbedder``), which groups events by what they
talk about rather than by their numeric stats.

Experiment design
-----------------
- Outlier screen (K-Means / Ward / GMM only): partitional and hierarchical
  methods are pulled apart by single extreme points (one viral story
  corroborated 52 times sits ~130 standard deviations from the mass in this
  dataset). Rows whose standardized-feature L2 norm exceeds 10x the median
  norm are screened out of THOSE experiments (recorded as
  ``excluded_indices``; labels for them are ``None``). DBSCAN is deliberately
  run on the FULL matrix — flagging such points as noise is exactly its job.
- K-Means: sweep k = 2..8 (random_state=42). A k is eligible only if every
  cluster holds >= 5 points (micro-clusters are outlier artifacts); the k
  with the best silhouette among eligible labelings wins, with the raw best
  kept as a flagged fallback if nothing is eligible.
- Agglomerative (Ward linkage): same sweep, eligibility rule and criterion.
- DBSCAN: eps from a deterministic k-distance knee — the distance to the
  ``min_samples``-th neighbour is computed for every point and sorted
  ascending; normalizing the curve to the unit square, the knee is the point
  with the largest (x - y) gap below the diagonal (the correct direction for
  this dataset's concave curve: a long gentle rise, then a steep outlier
  tail). min_samples = 5.
- Gaussian Mixture (EM): fitted with the K-Means-selected component count
  (random_state=42); clusters are the posterior-argmax assignments.
- PCA (2 components) of the standardized features gives the shared 2-D
  projection the label arrays are index-aligned with (points returned for
  plotting).

Silhouette is reported wherever computable (>= 2 clusters, fewer clusters
than points); for DBSCAN, noise points are excluded first.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from .text import LsaEmbedder

SEED = 42
K_RANGE = range(2, 9)
AGGLOMERATIVE_MAX_N = 3000   # ward is O(n^2) in memory beyond this
MIN_CLUSTER_SIZE = 5         # eligibility: no micro-clusters in the selection
OUTLIER_NORM_FACTOR = 10.0   # screen rows further than 10x the median norm


def _sizes(labels: Sequence[int]) -> dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {str(label): counts[label] for label in sorted(counts)}


def _silhouette(X: np.ndarray, labels: Sequence[int], drop_noise: bool = False) -> float | None:
    """Silhouette score, or None when not computable (needs 2..n-1 clusters)."""
    array = np.asarray(labels)
    if drop_noise:
        mask = array != -1
        if mask.sum() < 3:
            return None
        array = array[mask]
        matrix = X[mask]
    else:
        matrix = X
    n_clusters = len(set(array.tolist()))
    if n_clusters < 2 or n_clusters >= len(array):
        return None
    try:
        return round(float(silhouette_score(matrix, array, metric="euclidean")), 4)
    except ValueError:
        return None


def _outlier_mask(X: np.ndarray) -> np.ndarray:
    """Boolean mask of rows kept under the 10x-median-norm screen."""
    norms = np.linalg.norm(X, axis=1)
    median = float(np.median(norms))
    if median <= 0:
        return np.ones(len(X), dtype=bool)
    return norms <= OUTLIER_NORM_FACTOR * median


def _select_best_k(
    X: np.ndarray, labels_by_k: dict[int, np.ndarray]
) -> tuple[int | None, float | None, dict[str, float | None], bool]:
    """Best k among eligible (no micro-cluster) labelings, else flagged fallback."""
    silhouette_by_k: dict[str, float | None] = {}
    best_eligible: tuple[int, float] | None = None
    best_any: tuple[int, float] | None = None
    for k, labels in labels_by_k.items():
        score = _silhouette(X, labels)
        silhouette_by_k[str(k)] = score
        if score is None:
            continue
        if best_any is None or score > best_any[1]:
            best_any = (k, score)
        if len(labels) and min(np.bincount(labels)) >= MIN_CLUSTER_SIZE:
            if best_eligible is None or score > best_eligible[1]:
                best_eligible = (k, score)
    if best_eligible is not None:
        return best_eligible[0], best_eligible[1], silhouette_by_k, False
    if best_any is not None:
        return best_any[0], best_any[1], silhouette_by_k, True
    return None, None, silhouette_by_k, False


def _kmeans_experiment(X: np.ndarray) -> dict:
    """Sweep k = 2..8; return the eligibility-filtered best K-Means labels."""
    labels_by_k: dict[int, np.ndarray] = {}
    for k in K_RANGE:
        if k >= len(X):
            break
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels_by_k[k] = model.fit_predict(X)
    best_k, best_score, by_k, degenerate = _select_best_k(X, labels_by_k)
    if best_k is None:
        return {"best_k": None, "silhouette_by_k": by_k, "sizes": {},
                "silhouette": None, "labels": [], "inertia": None,
                "degenerate_selection": degenerate}
    return {
        "best_k": best_k,
        "silhouette_by_k": by_k,
        "silhouette": best_score,
        "sizes": _sizes(labels_by_k[best_k]),
        "labels": [int(label) for label in labels_by_k[best_k]],
        "degenerate_selection": degenerate,
    }


def _agglomerative_experiment(X: np.ndarray) -> dict:
    """Ward-linkage hierarchical clustering, same k sweep and eligibility rule."""
    sample_indices: list[int] | None = None
    matrix = X
    if len(X) > AGGLOMERATIVE_MAX_N:
        sample_indices = list(
            np.linspace(0, len(X) - 1, AGGLOMERATIVE_MAX_N, dtype=int)
        )
        matrix = X[sample_indices]
    labels_by_k: dict[int, np.ndarray] = {}
    for k in K_RANGE:
        if k >= len(matrix):
            break
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels_by_k[k] = model.fit_predict(matrix)
    best_k, best_score, by_k, degenerate = _select_best_k(matrix, labels_by_k)
    if best_k is None:
        return {"best_k": None, "silhouette_by_k": by_k, "sizes": {},
                "silhouette": None, "labels": [], "sample_indices": sample_indices,
                "subsampled": sample_indices is not None,
                "degenerate_selection": degenerate}
    return {
        "best_k": best_k,
        "silhouette_by_k": by_k,
        "silhouette": best_score,
        "sizes": _sizes(labels_by_k[best_k]),
        "labels": [int(label) for label in labels_by_k[best_k]],
        "sample_indices": sample_indices,
        "subsampled": sample_indices is not None,
        "degenerate_selection": degenerate,
    }


def _dbscan_eps(X: np.ndarray, min_samples: int = 5) -> float:
    """k-distance knee heuristic for this dataset's concave curve.

    The distance to the ``min_samples``-th nearest neighbour is computed for
    every point and sorted ascending. Normalizing the curve into the unit
    square, the knee is the point with the largest (x - y) gap below the
    diagonal — where the gently rising curve lifts off toward the steep
    outlier tail. (The mirrored max(y - x) rule would fire at index 0 here
    and collapse DBSCAN to all-noise.)
    """
    neighbours = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = neighbours.kneighbors(X)
    kdist = np.sort(distances[:, -1])
    if len(kdist) < 3:
        return float(np.mean(kdist)) if len(kdist) else 0.5
    x = np.linspace(0.0, 1.0, len(kdist))
    span = float(kdist[-1] - kdist[0])
    if span <= 1e-12:
        return float(kdist[0]) or 0.5
    y = (kdist - kdist[0]) / span
    knee = int(np.argmax(x - y))
    eps = float(kdist[knee])
    return round(eps, 4) if eps > 0 else 0.5


def _dbscan_experiment(X: np.ndarray) -> dict:
    eps = _dbscan_eps(X, min_samples=5)
    model = DBSCAN(eps=eps, min_samples=5)
    labels = model.fit_predict(X)
    labels_list = [int(label) for label in labels]
    noise_count = sum(1 for label in labels_list if label == -1)
    n_clusters = len(set(labels_list) - {-1})
    return {
        "eps": eps,
        "min_samples": 5,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "sizes": _sizes(labels_list),
        "silhouette": _silhouette(X, labels_list, drop_noise=True),
        "labels": labels_list,
    }


def _gmm_experiment(X: np.ndarray, k: int | None) -> dict:
    """EM / Gaussian Mixture with the K-Means-selected component count."""
    if k is None or k >= len(X):
        return {"k": None, "sizes": {}, "silhouette": None, "labels": [],
                "aic": None, "bic": None, "converged": None}
    model = GaussianMixture(n_components=k, random_state=SEED)
    labels = model.fit_predict(X)
    return {
        "k": k,
        "sizes": _sizes(labels),
        "silhouette": _silhouette(X, labels),
        "labels": [int(label) for label in labels],
        "aic": round(float(model.aic(X)), 4),
        "bic": round(float(model.bic(X)), 4),
        "converged": bool(model.converged_),
    }


def _pca_experiment(X: np.ndarray) -> dict:
    """2-component PCA projection shared by all clusterings of this matrix."""
    n_components = min(2, X.shape[1], max(len(X), 1))
    model = PCA(n_components=n_components)
    projected = model.fit_transform(X)
    points = [[round(float(a), 4), round(float(b), 4)] for a, b in projected]
    ratio = [round(float(value), 4) for value in model.explained_variance_ratio_]
    while len(ratio) < 2:
        ratio.append(0.0)
    return {"points": points, "explained_variance_ratio": ratio}


def cluster_numeric(X, feature_names: Sequence[str]) -> dict:
    """Run every numeric clustering experiment on the standardized matrix."""
    matrix = np.asarray(X, dtype=float)
    mask = _outlier_mask(matrix)
    filtered = matrix[mask]
    excluded_indices = [int(i) for i in np.flatnonzero(~mask)]

    kmeans = _kmeans_experiment(filtered)
    gmm = _gmm_experiment(filtered, kmeans["best_k"])

    # pad partitional label arrays back to full length, None at screened rows
    def align(labels: list[int]) -> list[int | None]:
        aligned: list[int | None] = [None] * len(matrix)
        iterator = iter(labels)
        for position, kept in enumerate(mask):
            if kept:
                aligned[position] = next(iterator)
        return aligned

    kmeans["labels"] = align(kmeans["labels"])
    gmm["labels"] = align(gmm["labels"])

    agglomerative = _agglomerative_experiment(filtered)
    if agglomerative["labels"]:
        agglomerative["labels"] = align(agglomerative["labels"])

    return {
        "n_samples": int(matrix.shape[0]),
        "feature_names": list(feature_names),
        "outlier_screen": {
            "rule": "exclude rows with standardized-norm > 10x median norm "
                    "(K-Means/Ward/GMM only; DBSCAN sees the full matrix)",
            "excluded_count": len(excluded_indices),
            "excluded_indices": excluded_indices,
        },
        "kmeans": kmeans,
        "agglomerative": agglomerative,
        "dbscan": _dbscan_experiment(matrix),
        "gmm": gmm,
        "pca_2d": _pca_experiment(matrix),
    }


def cluster_semantic(documents: Sequence[str]) -> dict:
    """K-Means over LSA text embeddings: the semantic event clustering.

    Embeds every headline with ``text.LsaEmbedder`` (deterministic subspace
    iteration) and sweeps K-Means k = 2..8 on the embeddings with the same
    eligibility rule. Also returns a 2-D PCA projection of the embedding
    space for plotting.
    """
    documents = list(documents)
    embedder = LsaEmbedder()
    embeddings = np.asarray(embedder.fit_transform(documents), dtype=float)
    if embeddings.size == 0:
        return {"n_samples": len(documents), "embedding_dim": 0,
                "kmeans": {"best_k": None, "silhouette_by_k": {}, "sizes": {},
                           "silhouette": None, "labels": []},
                "pca_2d": {"points": [], "explained_variance_ratio": [0.0, 0.0]}}
    kmeans = _kmeans_experiment(embeddings)
    return {
        "n_samples": len(documents),
        "embedding_dim": int(embeddings.shape[1]),
        "kmeans": kmeans,
        "pca_2d": _pca_experiment(embeddings),
    }
