"""Evaluation metrics for embeddings, clusters, and recommendation sanity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def cluster_distribution(labels: np.ndarray) -> pd.Series:
    """Return the number of songs in each cluster."""
    return pd.Series(labels, name="cluster").value_counts().sort_index()


def average_intra_cluster_distance(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Average point-to-centroid distance across non-empty clusters."""
    per_cluster_distances: list[float] = []

    for cluster_id in np.unique(labels):
        cluster_points = embeddings[labels == cluster_id]
        if len(cluster_points) < 2:
            continue
        centroid = cluster_points.mean(axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        per_cluster_distances.append(float(distances.mean()))

    return float(np.mean(per_cluster_distances)) if per_cluster_distances else float("nan")


def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute silhouette score when enough clusters exist."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(embeddings):
        return None
    return float(silhouette_score(embeddings, labels))


def embedding_distance_correlation(
    feature_matrix: np.ndarray,
    embeddings: np.ndarray,
    sample_size: int = 250,
    random_state: int = 42,
) -> float | None:
    """Correlation between pairwise distances in feature space and embedding space."""
    if len(feature_matrix) < 3:
        return None

    rng = np.random.default_rng(random_state)
    selected_indices = rng.choice(
        len(feature_matrix),
        size=min(sample_size, len(feature_matrix)),
        replace=False,
    )
    sampled_features = feature_matrix[selected_indices]
    sampled_embeddings = embeddings[selected_indices]

    feature_distances = np.linalg.norm(sampled_features[:, None, :] - sampled_features[None, :, :], axis=2)
    embedding_distances = np.linalg.norm(sampled_embeddings[:, None, :] - sampled_embeddings[None, :, :], axis=2)
    upper_triangle_mask = np.triu(np.ones_like(feature_distances, dtype=bool), k=1)

    feature_values = feature_distances[upper_triangle_mask]
    embedding_values = embedding_distances[upper_triangle_mask]
    if np.std(feature_values) == 0.0 or np.std(embedding_values) == 0.0:
        return None

    correlation_matrix = np.corrcoef(feature_values, embedding_values)
    return float(correlation_matrix[0, 1])


def embedding_neighbor_overlap(
    feature_matrix: np.ndarray,
    embeddings: np.ndarray,
    k: int = 5,
    sample_size: int = 100,
    random_state: int = 42,
) -> float | None:
    """Estimate whether local nearest-neighbor structure is preserved by the embeddings."""
    if len(feature_matrix) <= k + 1:
        return None

    rng = np.random.default_rng(random_state)
    selected_indices = rng.choice(
        len(feature_matrix),
        size=min(sample_size, len(feature_matrix)),
        replace=False,
    )

    overlaps: list[float] = []
    for query_index in selected_indices:
        feature_distances = np.linalg.norm(feature_matrix - feature_matrix[query_index], axis=1)
        embedding_distances = np.linalg.norm(embeddings - embeddings[query_index], axis=1)

        feature_neighbors = np.argsort(feature_distances)[1 : k + 1]
        embedding_neighbors = np.argsort(embedding_distances)[1 : k + 1]
        union = set(feature_neighbors) | set(embedding_neighbors)
        if not union:
            continue
        overlap = len(set(feature_neighbors) & set(embedding_neighbors)) / len(union)
        overlaps.append(overlap)

    return float(np.mean(overlaps)) if overlaps else None


def summarize_pipeline_metrics(feature_matrix: np.ndarray, embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
    """Return a compact metric summary for the current pipeline state."""
    return {
        "cluster_count": int(len(np.unique(labels))),
        "average_intra_cluster_distance": average_intra_cluster_distance(embeddings, labels),
        "silhouette_score": compute_silhouette(embeddings, labels),
        "distance_correlation": embedding_distance_correlation(feature_matrix, embeddings),
        "neighbor_overlap": embedding_neighbor_overlap(feature_matrix, embeddings),
    }
