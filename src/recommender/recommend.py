"""Recommendation engine built on top of song embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.recommender.similarity import top_k_similar_indices


@dataclass(slots=True)
class RecommendationEngine:
    """Provide search and recommendation queries over an in-memory song catalog."""

    catalog: pd.DataFrame
    embeddings: np.ndarray
    cluster_labels: np.ndarray | None = None

    def song_label(self, index: int) -> str:
        row = self.catalog.loc[index]
        return f"{row['song_name']} - {row['artist_name']}"

    def search_songs(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Return songs matching a case-insensitive name or artist substring."""
        normalized_query = query.strip()
        if not normalized_query:
            return self.catalog.head(limit)

        mask = self.catalog["song_name"].str.contains(normalized_query, case=False, na=False)
        mask |= self.catalog["artist_name"].str.contains(normalized_query, case=False, na=False)
        return self.catalog.loc[mask].head(limit)

    def get_song_row(self, song_query: str | int) -> pd.Series:
        """Resolve a song query and return the catalog row."""
        return self.catalog.loc[self._resolve_song_index(song_query)]

    def recommend_by_embedding(self, song_query: str | int, top_k: int = 10) -> pd.DataFrame:
        """Recommend the nearest neighbors in embedding space."""
        query_index = self._resolve_song_index(song_query)
        indices, scores = top_k_similar_indices(self.embeddings, query_index, top_k=top_k)
        return self._build_result_frame(indices, scores)

    def recommend_by_cluster(self, song_query: str | int, top_k: int = 10) -> pd.DataFrame:
        """Recommend similar songs while staying inside the same cluster when possible."""
        query_index = self._resolve_song_index(song_query)
        if self.cluster_labels is None:
            return self.recommend_by_embedding(query_index, top_k=top_k)

        query_cluster = int(self.cluster_labels[query_index])
        candidate_indices = np.flatnonzero(self.cluster_labels == query_cluster)
        indices, scores = top_k_similar_indices(
            self.embeddings,
            query_index=query_index,
            top_k=top_k,
            candidate_indices=candidate_indices,
        )

        if len(indices) == 0:
            return self.recommend_by_embedding(query_index, top_k=top_k)
        return self._build_result_frame(indices, scores)

    def _resolve_song_index(self, song_query: str | int) -> int:
        if isinstance(song_query, (int, np.integer)):
            return int(song_query)

        normalized_query = song_query.strip().lower()
        label_matches = self.catalog.apply(
            lambda row: f"{row['song_name']} - {row['artist_name']}".lower() == normalized_query,
            axis=1,
        )
        if label_matches.any():
            return int(label_matches[label_matches].index[0])

        exact_name_matches = self.catalog["song_name"].str.lower() == normalized_query
        if exact_name_matches.any():
            return int(exact_name_matches[exact_name_matches].index[0])

        contains_matches = self.catalog["song_name"].str.contains(normalized_query, case=False, na=False)
        if contains_matches.any():
            return int(contains_matches[contains_matches].index[0])

        raise KeyError(f"Could not find a song matching query: {song_query}")

    def _build_result_frame(self, indices: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
        if len(indices) == 0:
            return pd.DataFrame(columns=["song_name", "artist_name", "playlist_genre", "cluster", "similarity_score"])

        result_frame = self.catalog.loc[indices].copy()
        result_frame["similarity_score"] = scores
        if self.cluster_labels is not None and "cluster" not in result_frame.columns:
            result_frame["cluster"] = self.cluster_labels[indices]
        ordered_columns = [
            column
            for column in ["song_name", "artist_name", "album_name", "playlist_genre", "cluster", "popularity", "similarity_score"]
            if column in result_frame.columns
        ]
        return result_frame[ordered_columns].reset_index(drop=True)
