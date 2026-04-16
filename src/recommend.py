"""KNN-based song recommendation with fuzzy search."""

from __future__ import annotations
from unittest import result

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.neighbors import NearestNeighbors

from src.features import FEATURE_COLUMNS_ENCODED


class RecommendationEngine:
    """Song-to-song recommendation using cosine similarity in feature space."""

    def __init__(self, df: pd.DataFrame, feature_matrix: np.ndarray, k_neighbors: int = 100) -> None:
        self.df = df.reset_index(drop=True)
        self.feature_matrix = feature_matrix
        self.k_neighbors = min(k_neighbors, len(df) - 1)

        self.nn = NearestNeighbors(
            n_neighbors=self.k_neighbors,
            metric="cosine",
            algorithm="brute",
        )
        self.nn.fit(self.feature_matrix)

    # def search_songs(self, query: str, limit: int = 10) -> pd.DataFrame:
    #     """Fuzzy search for songs by track_name or artists."""
    #     if not query.strip():
    #         return self.df.head(limit)

    #     labels = (self.df["track_name"] + " - " + self.df["artists"]).tolist()
    #     matches = process.extract(query, labels, scorer=fuzz.WRatio, limit=limit)
    #     indices = [m[2] for m in matches if m[1] >= 40]

    #     if not indices:
    #         return pd.DataFrame(columns=self.df.columns)

    #     return self.df.iloc[indices][["track_name", "artists", "album_name", "track_genre", "popularity"]].copy()

    def search_songs(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Search songs with simple ranking: title prefix > title contains > artist contains > fuzzy fallback."""
        normalized_query = query.strip().lower()
        if not normalized_query:
            return self.df.head(limit)[["track_name", "artists", "album_name", "track_genre", "popularity"]].copy()

        df = self.df.copy()

        title_lower = df["track_name"].fillna("").str.lower()
        artist_lower = df["artists"].fillna("").str.lower()

        # 1. exact title prefix
        prefix_matches = df[title_lower.str.startswith(normalized_query)]

        # 2. title contains
        contains_matches = df[title_lower.str.contains(normalized_query, na=False)]

        # 3. artist contains
        artist_matches = df[artist_lower.str.contains(normalized_query, na=False)]

        # combine in priority order, remove duplicates
        combined = pd.concat([prefix_matches, contains_matches, artist_matches]).drop_duplicates()

        # if enough results, return top ones directly
        if len(combined) >= limit:
            result = combined.head(limit)[["track_name", "artists", "album_name", "track_genre", "popularity"]].copy()
            return result.reset_index(drop=True)

        # 4. fuzzy fallback for remaining slots
        labels = (df["track_name"] + " - " + df["artists"]).tolist()
        matches = process.extract(query, labels, scorer=fuzz.WRatio, limit=limit * 2)
        fuzzy_indices = [m[2] for m in matches if m[1] >= 60]

        fuzzy_df = df.iloc[fuzzy_indices]
        final = pd.concat([combined, fuzzy_df]).drop_duplicates().head(limit)

        result = final[["track_name", "artists", "album_name", "track_genre", "popularity"]].copy()
        return result.reset_index(drop=True)
    
    # 1 playlist
    def _filter_same_name(self, song_index: int, indices: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remove songs with the same track_name as the query (duplicate versions)."""
        query_name = self.df.iloc[song_index]["track_name"]
        keep = []
        for i, idx in enumerate(indices):
            if self.df.iloc[idx]["track_name"] != query_name:
                keep.append(i)
        keep = np.array(keep, dtype=int)
        if len(keep) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        return indices[keep], distances[keep]
    

    def _combine_query_vectors(self, song_indices: list[int]) -> np.ndarray:
        """Combine multiple seed songs into one query vector by averaging."""
        vectors = self.feature_matrix[song_indices]
        return vectors.mean(axis=0)
    

    def recommend_from_playlist(self, song_indices: list[int], top_k: int = 10) -> pd.DataFrame:
        """Recommend songs based on multiple seed songs."""
        if len(song_indices) == 0:
            return pd.DataFrame(columns=["track_name", "artists", "track_genre", "popularity", "similarity"])

        query_vector = self._combine_query_vectors(song_indices).reshape(1, -1)
        distances, indices = self.nn.kneighbors(query_vector, n_neighbors=self.k_neighbors)

        distances = distances.flatten()
        indices = indices.flatten()

        # Remove the selected seed songs themselves
        seed_set = set(song_indices)
        keep = [i for i, idx in enumerate(indices) if idx not in seed_set]
        keep = np.array(keep, dtype=int)

        if len(keep) == 0:
            return pd.DataFrame(columns=["track_name", "artists", "track_genre", "popularity", "similarity"])

        indices = indices[keep]
        distances = distances[keep]

        # Remove songs with the same track_name as any seed song
        seed_names = set(self.df.iloc[song_indices]["track_name"].tolist())
        keep = [i for i, idx in enumerate(indices) if self.df.iloc[idx]["track_name"] not in seed_names]
        keep = np.array(keep, dtype=int)

        if len(keep) == 0:
            return pd.DataFrame(columns=["track_name", "artists", "track_genre", "popularity", "similarity"])

        indices = indices[keep][:top_k]
        distances = distances[keep][:top_k]

        similarities = 1 - distances

        result = self.df.iloc[indices][["track_name", "artists", "track_genre", "popularity"]].copy()
        result["similarity"] = similarities
        result = result.reset_index(drop=True)
        return result

    def recommend(self, song_index: int, top_k: int = 10) -> pd.DataFrame:
        """Return top-K most similar songs (excluding same-name duplicates)."""
        query_vector = self.feature_matrix[song_index].reshape(1, -1)
        distances, indices = self.nn.kneighbors(query_vector, n_neighbors=self.k_neighbors)

        distances = distances.flatten()
        indices = indices.flatten()

        # Remove query song itself
        mask = indices != song_index
        indices = indices[mask]
        distances = distances[mask]

        # Remove same-name duplicates
        indices, distances = self._filter_same_name(song_index, indices, distances)

        indices = indices[:top_k]
        distances = distances[:top_k]
        similarities = 1 - distances

        result = self.df.iloc[indices][["track_name", "artists", "track_genre", "popularity"]].copy()
        result["similarity"] = similarities
        result = result.reset_index(drop=True)
        return result

    def recommend_with_features(self, song_index: int, top_k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return recommendations along with a feature comparison DataFrame."""
        query_vector = self.feature_matrix[song_index].reshape(1, -1)
        distances, indices = self.nn.kneighbors(query_vector, n_neighbors=self.k_neighbors)

        distances = distances.flatten()
        indices = indices.flatten()

        mask = indices != song_index
        indices = indices[mask]
        distances = distances[mask]

        indices, distances = self._filter_same_name(song_index, indices, distances)

        indices = indices[:top_k]
        distances = distances[:top_k]
        similarities = 1 - distances

        recs = self.df.iloc[indices][["track_name", "artists", "track_genre", "popularity"]].copy()
        recs["similarity"] = similarities
        recs = recs.reset_index(drop=True)

        # Build feature comparison
        query_features = self.df.iloc[song_index][FEATURE_COLUMNS_ENCODED].to_dict()
        query_features["track_name"] = self.df.iloc[song_index]["track_name"]
        query_features["role"] = "Query"

        rows = [query_features]
        for idx in indices:
            row = self.df.iloc[idx][FEATURE_COLUMNS_ENCODED].to_dict()
            row["track_name"] = self.df.iloc[idx]["track_name"]
            row["role"] = "Recommended"
            rows.append(row)

        feature_comparison = pd.DataFrame(rows)
        return recs, feature_comparison

    # ── Cluster-aware recommendation (Phase 2) ─────────────────────────────

    def recommend_by_cluster(
        self,
        song_index: int,
        cluster_labels: np.ndarray,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Recommend within the same K-Means cluster, ranked by cosine similarity."""
        query_cluster = int(cluster_labels[song_index])
        candidate_mask = cluster_labels == query_cluster

        # Get candidates in the same cluster
        candidate_indices = np.flatnonzero(candidate_mask)
        candidate_indices = candidate_indices[candidate_indices != song_index]

        # Filter same-name duplicates
        query_name = self.df.iloc[song_index]["track_name"]
        candidate_indices = np.array(
            [i for i in candidate_indices if self.df.iloc[i]["track_name"] != query_name],
            dtype=int,
        )

        if len(candidate_indices) == 0:
            # Fall back to full-catalog recommendation
            return self.recommend(song_index, top_k=top_k)

        # Compute cosine similarity for candidates
        query_vec = self.feature_matrix[song_index]
        candidate_vecs = self.feature_matrix[candidate_indices]
        dots = candidate_vecs @ query_vec
        query_norm = np.linalg.norm(query_vec)
        cand_norms = np.linalg.norm(candidate_vecs, axis=1)
        safe_denom = np.where(cand_norms * query_norm > 0, cand_norms * query_norm, 1.0)
        similarities = dots / safe_denom

        # Top-K
        ranked = np.argsort(similarities)[::-1][:top_k]
        top_indices = candidate_indices[ranked]
        top_sims = similarities[ranked]

        result = self.df.iloc[top_indices][["track_name", "artists", "track_genre", "popularity"]].copy()
        result["similarity"] = top_sims
        result = result.reset_index(drop=True)
        return result

    def recommend_by_gmm(
        self,
        song_index: int,
        probabilities: np.ndarray,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Recommend by cosine similarity of GMM posterior probability vectors."""
        query_prob = probabilities[song_index]
        query_name = self.df.iloc[song_index]["track_name"]

        # Cosine similarity between query's posterior and all others
        dots = probabilities @ query_prob
        query_norm = np.linalg.norm(query_prob)
        all_norms = np.linalg.norm(probabilities, axis=1)
        safe_denom = np.where(all_norms * query_norm > 0, all_norms * query_norm, 1.0)
        similarities = dots / safe_denom

        # Exclude self and same-name songs
        similarities[song_index] = -np.inf
        for i in range(len(self.df)):
            if self.df.iloc[i]["track_name"] == query_name and i != song_index:
                similarities[i] = -np.inf

        ranked = np.argsort(similarities)[::-1][:top_k]
        top_sims = similarities[ranked]

        result = self.df.iloc[ranked][["track_name", "artists", "track_genre", "popularity"]].copy()
        result["similarity"] = top_sims
        result = result.reset_index(drop=True)
        return result

    def song_label(self, index: int) -> str:
        """Human-readable label for a song."""
        row = self.df.iloc[index]
        return f"{row['track_name']} - {row['artists']}"

def rerank_feature_auto(
    engine: RecommendationEngine,
    recs: pd.DataFrame,
    seed_index: int,
    alpha: float = 0.15,
    ) -> pd.DataFrame:
    """Simple feature-aware reranking using distance to the query song vector."""
    if recs.empty:
        return recs

    query_vec = engine.feature_matrix[seed_index]
    rerank_scores = []

    for i in range(len(recs)):
        rec_name = recs.iloc[i]["track_name"]
        rec_artist = recs.iloc[i]["artists"]

        match = engine.df[
            (engine.df["track_name"] == rec_name) &
            (engine.df["artists"] == rec_artist)
        ]

        if len(match) == 0:
            rerank_scores.append(recs.iloc[i]["similarity"])
            continue

        rec_idx = match.index[0]
        rec_vec = engine.feature_matrix[rec_idx]

        feature_dist = np.linalg.norm(rec_vec - query_vec)
        new_score = float(recs.iloc[i]["similarity"]) - alpha * feature_dist
        rerank_scores.append(new_score)

    reranked = recs.copy()
    reranked["rerank_score"] = rerank_scores
    reranked = reranked.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    return reranked
