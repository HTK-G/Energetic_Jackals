"""Clustering algorithms (K-Means, GMM) with hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from src.custom_kmeans import CustomKMeans
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


@dataclass(slots=True)
class ClusterResult:
    """Container for a fitted clustering model's outputs."""

    algorithm: str
    labels: np.ndarray
    n_clusters: int
    model: CustomKMeans | GaussianMixture
    # GMM-specific: soft posterior probabilities (n_songs, n_clusters)
    probabilities: np.ndarray | None = None


@dataclass(slots=True)
class TuningResult:
    """Hyperparameter search results for choosing optimal K."""

    k_range: list[int]
    inertias: list[float] = field(default_factory=list)        # K-Means only
    silhouette_scores: list[float] = field(default_factory=list)
    bics: list[float] = field(default_factory=list)            # GMM only
    best_k: int = 0


# ── K-Means ──────────────────────────────────────────────────────────────────


def tune_kmeans(
    feature_matrix: np.ndarray,
    k_range: range = range(5, 31),
    random_state: int = 42,
) -> TuningResult:
    """Run K-Means over a range of K and collect inertia + silhouette scores.

    Uses sklearn's KMeans for tuning (much faster than CustomKMeans). The final
    model is later refit with CustomKMeans by `fit_kmeans`, satisfying the
    course requirement that the deployed model use a from-scratch implementation.
    """
    result = TuningResult(k_range=list(k_range))

    for k in k_range:
        km = SKLearnKMeans(n_clusters=k, random_state=random_state, n_init=1)
        labels = km.fit_predict(feature_matrix)
        result.inertias.append(float(km.inertia_))
        result.silhouette_scores.append(float(
            silhouette_score(feature_matrix, labels, sample_size=5000, random_state=random_state)
        ))

    best_idx = int(np.argmax(result.silhouette_scores))
    result.best_k = result.k_range[best_idx]
    return result


def fit_kmeans(
    feature_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> ClusterResult:
    """Fit K-Means with a given K and return results."""
    km = CustomKMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(feature_matrix)
    return ClusterResult(
        algorithm="K-Means",
        labels=labels,
        n_clusters=n_clusters,
        model=km,
    )


# ── GMM ──────────────────────────────────────────────────────────────────────


def tune_gmm(
    feature_matrix: np.ndarray,
    k_range: range = range(5, 31),
    covariance_type: str = "full",
    random_state: int = 42,
    n_init: int = 1,
) -> TuningResult:
    """Run GMM over a range of K and collect BIC scores. K selected by min BIC."""
    result = TuningResult(k_range=list(k_range))

    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=n_init,
        )
        gmm.fit(feature_matrix)
        result.bics.append(float(gmm.bic(feature_matrix)))

    best_idx = int(np.argmin(result.bics))
    result.best_k = result.k_range[best_idx]
    return result



def fit_gmm(
    feature_matrix: np.ndarray,
    n_clusters: int,
    covariance_type: str = "full",
    random_state: int = 42,
    n_init: int = 3,
) -> ClusterResult:
    """Fit GMM with a given K and return results with soft probabilities."""
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
    )
    labels = gmm.fit_predict(feature_matrix)
    probabilities = gmm.predict_proba(feature_matrix)
    return ClusterResult(
        algorithm=f"GMM ({covariance_type})",
        labels=labels,
        n_clusters=n_clusters,
        model=gmm,
        probabilities=probabilities,
    )