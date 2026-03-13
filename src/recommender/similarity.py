"""Similarity functions used by the recommendation engine."""

from __future__ import annotations

import numpy as np


def cosine_similarity_scores(query_vector: np.ndarray, candidate_matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and candidate vectors."""
    query_vector = np.asarray(query_vector, dtype=float)
    candidate_matrix = np.asarray(candidate_matrix, dtype=float)

    query_norm = float(np.linalg.norm(query_vector))
    candidate_norms = np.linalg.norm(candidate_matrix, axis=1)

    safe_query_norm = query_norm if query_norm > 0.0 else 1.0
    safe_candidate_norms = np.where(candidate_norms > 0.0, candidate_norms, 1.0)
    return (candidate_matrix @ query_vector) / (safe_candidate_norms * safe_query_norm)


def top_k_similar_indices(
    embeddings: np.ndarray,
    query_index: int,
    top_k: int = 10,
    candidate_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices and scores for the most similar songs to the query song."""
    embeddings = np.asarray(embeddings, dtype=float)
    available_indices = (
        np.asarray(candidate_indices, dtype=int)
        if candidate_indices is not None
        else np.arange(len(embeddings), dtype=int)
    )

    available_indices = available_indices[available_indices != query_index]
    if len(available_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    scores = cosine_similarity_scores(embeddings[query_index], embeddings[available_indices])
    ranked_positions = np.argsort(scores)[::-1][:top_k]
    return available_indices[ranked_positions], scores[ranked_positions]
