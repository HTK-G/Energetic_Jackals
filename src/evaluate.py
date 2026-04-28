"""Evaluation metrics: genre hit rate, internal cluster metrics, external cluster metrics."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from src.recommend import RecommendationEngine


# ── Genre hit rate (Phase 1) ─────────────────────────────────────────────────


def parse_genre_set(genre_str: str) -> set[str]:
    """Parse the all_genres column (string repr of a list) into a set."""
    try:
        genres = ast.literal_eval(genre_str)
        if isinstance(genres, list):
            return {g.strip().lower() for g in genres}
    except (ValueError, SyntaxError):
        pass
    return {genre_str.strip().lower()} if isinstance(genre_str, str) and genre_str.strip() else set()


def genre_hit_rate(
    engine: RecommendationEngine,
    song_index: int,
    top_k: int = 10,
) -> float:
    """Fraction of top-K recommendations sharing at least one genre with the query."""
    query_genres = parse_genre_set(engine.df.iloc[song_index].get("all_genres", ""))
    if not query_genres:
        query_genres = {engine.df.iloc[song_index]["track_genre"].strip().lower()}

    recs = engine.recommend(song_index, top_k=top_k)
    if recs.empty:
        return 0.0

    query_vector = engine.feature_matrix[song_index].reshape(1, -1)
    distances, indices = engine.nn.kneighbors(query_vector, n_neighbors=min(top_k + 1, engine.k_neighbors))
    indices = indices.flatten()
    indices = indices[indices != song_index][:top_k]

    hits = 0
    for idx in indices:
        rec_genres = parse_genre_set(engine.df.iloc[idx].get("all_genres", ""))
        if not rec_genres:
            rec_genres = {engine.df.iloc[idx]["track_genre"].strip().lower()}
        if query_genres & rec_genres:
            hits += 1

    return hits / len(indices) if len(indices) > 0 else 0.0


def average_genre_hit_rate(
    engine: RecommendationEngine,
    top_k: int = 10,
    sample_size: int = 200,
    random_state: int = 42,
) -> float:
    """Average genre hit rate across a random sample of query songs."""
    rng = np.random.default_rng(random_state)
    n = len(engine.df)
    sample_indices = rng.choice(n, size=min(sample_size, n), replace=False)
    rates = [genre_hit_rate(engine, int(idx), top_k=top_k) for idx in sample_indices]
    return float(np.mean(rates))


# ── Internal cluster metrics (Phase 2) ───────────────────────────────────────


@dataclass(slots=True)
class ClusterMetrics:
    """All evaluation metrics for a single clustering result."""

    algorithm: str
    n_clusters: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    ari: float | None = None
    nmi: float | None = None


def compute_internal_metrics(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz scores.

    Silhouette uses sample_size=5000 because it has O(n^2) complexity and the
    full 89K computation is too slow. davies_bouldin and calinski_harabasz are
    O(n) and use the full dataset.
    """
    n_unique = len(np.unique(labels))
    if n_unique < 2 or n_unique >= len(feature_matrix):
        return {"silhouette": 0.0, "davies_bouldin": float("inf"), "calinski_harabasz": 0.0}

    return {
        "silhouette": float(silhouette_score(
            feature_matrix, labels, sample_size=5000, random_state=42
        )),
        "davies_bouldin": float(davies_bouldin_score(feature_matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(feature_matrix, labels)),
    }


def compute_external_metrics(
    labels: np.ndarray,
    genre_labels: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute ARI and NMI against genre ground truth."""
    return {
        "ari": float(adjusted_rand_score(genre_labels, labels)),
        "nmi": float(normalized_mutual_info_score(genre_labels, labels)),
    }


def evaluate_clustering(
    algorithm: str,
    n_clusters: int,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    genre_labels: np.ndarray | pd.Series | None = None,
) -> ClusterMetrics:
    """Full evaluation of a clustering result (internal + optional external)."""
    internal = compute_internal_metrics(feature_matrix, labels)
    metrics = ClusterMetrics(
        algorithm=algorithm,
        n_clusters=n_clusters,
        silhouette=internal["silhouette"],
        davies_bouldin=internal["davies_bouldin"],
        calinski_harabasz=internal["calinski_harabasz"],
    )
    if genre_labels is not None:
        external = compute_external_metrics(labels, genre_labels)
        metrics.ari = external["ari"]
        metrics.nmi = external["nmi"]
    return metrics


def metrics_comparison_table(metrics_list: list[ClusterMetrics]) -> pd.DataFrame:
    """Build a side-by-side comparison table of all clustering evaluations."""
    rows = []
    for m in metrics_list:
        rows.append({
            "Algorithm": m.algorithm,
            "K": m.n_clusters,
            "Silhouette": round(m.silhouette, 4),
            "Davies-Bouldin": round(m.davies_bouldin, 4),
            "Calinski-Harabasz": round(m.calinski_harabasz, 1),
            "ARI": round(m.ari, 4) if m.ari is not None else np.nan,
            "NMI": round(m.nmi, 4) if m.nmi is not None else np.nan,
        })
    return pd.DataFrame(rows)