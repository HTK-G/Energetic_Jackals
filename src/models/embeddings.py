"""Embedding helpers for PCA and autoencoder-based representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from src.models.autoencoder import AutoencoderConfig, AutoencoderEmbeddingModel


@dataclass(slots=True)
class EmbeddingResult:
    """Embeddings together with the fitted model and a 2D projection."""

    embeddings: np.ndarray
    method: str
    model: Any
    visualization_projection: np.ndarray


def project_embeddings_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings into two dimensions for plotting."""
    embeddings = np.asarray(embeddings, dtype=float)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if embeddings.shape[1] == 1:
        return np.column_stack([embeddings[:, 0], np.zeros(len(embeddings))])
    if embeddings.shape[1] == 2:
        return embeddings.copy()

    projector = PCA(n_components=2, random_state=42)
    return projector.fit_transform(embeddings)


def compute_pca_embeddings(feature_matrix: np.ndarray, embedding_dim: int = 12) -> tuple[np.ndarray, PCA]:
    """Learn PCA embeddings from the preprocessed feature matrix."""
    max_components = min(feature_matrix.shape[0], feature_matrix.shape[1], embedding_dim)
    pca = PCA(n_components=max(2, max_components), random_state=42)
    embeddings = pca.fit_transform(feature_matrix)
    return embeddings, pca


def compute_song_embeddings(
    feature_matrix: np.ndarray,
    method: str = "pca",
    embedding_dim: int = 12,
    autoencoder_epochs: int = 20,
) -> EmbeddingResult:
    """Train the configured embedding model and return embeddings with artifacts."""
    normalized_method = method.strip().lower()

    if normalized_method == "pca":
        embeddings, model = compute_pca_embeddings(feature_matrix, embedding_dim=embedding_dim)
    elif normalized_method == "autoencoder":
        config = AutoencoderConfig(
            input_dim=feature_matrix.shape[1],
            embedding_dim=min(embedding_dim, feature_matrix.shape[1]),
            epochs=autoencoder_epochs,
        )
        model = AutoencoderEmbeddingModel(config)
        embeddings = model.fit_transform(feature_matrix)
    else:
        raise ValueError(f"Unsupported embedding method: {method}")

    visualization_projection = project_embeddings_2d(embeddings)
    return EmbeddingResult(
        embeddings=np.asarray(embeddings, dtype=float),
        method=normalized_method,
        model=model,
        visualization_projection=visualization_projection,
    )
