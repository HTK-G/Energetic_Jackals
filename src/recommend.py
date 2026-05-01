"""KNN-based song recommendation with fuzzy search."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.neighbors import NearestNeighbors

from src.features import FEATURE_COLUMNS_ENCODED


def _dedupe_recs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate songs by (track_name, artists), keeping highest popularity.

    The catalog contains the same track on multiple track_ids (re-releases,
    explicit/clean variants, deluxe albums).  Without this step, a single
    recommendation list can show the same song two or three times.
    """
    if df.empty:
        return df
    return (
        df.sort_values("popularity", ascending=False, kind="stable")
          .drop_duplicates(subset=["track_name", "artists"], keep="first")
          .sort_index()
          .reset_index(drop=True)
    )


class RecommendationEngine:
    """Song-to-song recommendation using cosine similarity in feature space."""

    def __init__(
        self, df: pd.DataFrame, feature_matrix: np.ndarray, k_neighbors: int = 100
    ) -> None:
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
            return self.df.head(limit)[
                ["track_name", "artists", "album_name", "track_genre", "popularity"]
            ].copy()

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
        combined = pd.concat(
            [prefix_matches, contains_matches, artist_matches]
        ).drop_duplicates()

        # if enough results, return top ones directly
        if len(combined) >= limit:
            result = combined.head(limit)[
                ["track_name", "artists", "album_name", "track_genre", "popularity"]
            ].copy()
            return result.reset_index(drop=True)

        # 4. fuzzy fallback for remaining slots
        # Fill NaNs to avoid producing non-string candidates.
        labels = (
            df["track_name"].fillna("") + " - " + df["artists"].fillna("")
        ).tolist()
        matches = process.extract(query, labels, scorer=fuzz.WRatio, limit=limit * 2)
        fuzzy_indices = [m[2] for m in matches if m[1] >= 60]

        fuzzy_df = df.iloc[fuzzy_indices]
        final = pd.concat([combined, fuzzy_df]).drop_duplicates().head(limit)

        result = final[
            ["track_name", "artists", "album_name", "track_genre", "popularity"]
        ].copy()
        return result.reset_index(drop=True)

    def _filter_same_name(
        self, song_index: int, indices: np.ndarray, distances: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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
        empty_cols = ["track_name", "artists", "track_genre", "popularity", "similarity"]
        if len(song_indices) == 0:
            return pd.DataFrame(columns=empty_cols)

        query_vector = self._combine_query_vectors(song_indices).reshape(1, -1)
        distances, indices = self.nn.kneighbors(query_vector, n_neighbors=self.k_neighbors)

        distances = distances.flatten()
        indices = indices.flatten()

        # Remove the selected seed songs themselves
        seed_set = set(song_indices)
        keep = [i for i, idx in enumerate(indices) if idx not in seed_set]
        keep = np.array(keep, dtype=int)

        if len(keep) == 0:
            return pd.DataFrame(columns=empty_cols)

        indices = indices[keep]
        distances = distances[keep]

        # Remove songs with the same track_name as any seed song
        seed_names = set(self.df.iloc[song_indices]["track_name"].tolist())
        keep = [i for i, idx in enumerate(indices) if self.df.iloc[idx]["track_name"] not in seed_names]
        keep = np.array(keep, dtype=int)

        if len(keep) == 0:
            return pd.DataFrame(columns=empty_cols)

        # Over-fetch a generous slice so post-dedup we still have top_k
        fetch = min(top_k * 3, len(keep))
        indices = indices[keep][:fetch]
        distances = distances[keep][:fetch]
        similarities = 1 - distances

        result = self.df.iloc[indices][
            ["track_id", "track_name", "artists", "album_name", "track_genre", "popularity"]
        ].copy()
        result["similarity"] = similarities
        result = result.reset_index(drop=True)
        return _dedupe_recs(result).head(top_k).reset_index(drop=True)

    def recommend(self, song_index: int, top_k: int = 10) -> pd.DataFrame:
        """Return top-K most similar songs (excluding same-name duplicates)."""
        query_vector = self.feature_matrix[song_index].reshape(1, -1)
        distances, indices = self.nn.kneighbors(
            query_vector, n_neighbors=self.k_neighbors
        )

        distances = distances.flatten()
        indices = indices.flatten()

        # Remove query song itself
        mask = indices != song_index
        indices = indices[mask]
        distances = distances[mask]

        # Remove same-name duplicates of the query
        indices, distances = self._filter_same_name(song_index, indices, distances)

        # Over-fetch so post-dedup we still have top_k
        fetch = min(top_k * 3, len(indices))
        indices = indices[:fetch]
        distances = distances[:fetch]
        similarities = 1 - distances

        result = self.df.iloc[indices][
            ["track_id", "track_name", "artists", "album_name", "track_genre", "popularity"]
        ].copy()
        result["similarity"] = similarities
        result = result.reset_index(drop=True)
        return _dedupe_recs(result).head(top_k).reset_index(drop=True)

    def recommend_with_features(
        self, song_index: int, top_k: int = 10
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return recommendations along with a feature comparison DataFrame."""
        recs = self.recommend(song_index, top_k=top_k)

        query_features = self.df.iloc[song_index][FEATURE_COLUMNS_ENCODED].to_dict()
        query_features["track_name"] = self.df.iloc[song_index]["track_name"]
        query_features["artists"] = self.df.iloc[song_index]["artists"]
        query_features["role"] = "Query"

        rows = [query_features]
        for _, rec in recs.iterrows():
            match = self.df[
                (self.df["track_name"] == rec["track_name"])
                & (self.df["artists"] == rec["artists"])
            ]
            if len(match) == 0:
                continue
            row = match.iloc[0][FEATURE_COLUMNS_ENCODED].to_dict()
            row["track_name"] = rec["track_name"]
            row["artists"] = rec["artists"]
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
            [
                i
                for i in candidate_indices
                if self.df.iloc[i]["track_name"] != query_name
            ],
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

        # Over-fetch so post-dedup we still have top_k
        fetch = min(top_k * 3, len(similarities))
        ranked = np.argsort(similarities)[::-1][:fetch]
        top_indices = candidate_indices[ranked]
        top_sims = similarities[ranked]

        result = self.df.iloc[top_indices][
            ["track_id", "track_name", "artists", "album_name", "track_genre", "popularity"]
        ].copy()
        result["similarity"] = top_sims
        result = result.reset_index(drop=True)
        return _dedupe_recs(result).head(top_k).reset_index(drop=True)

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

        # Exclude self and same-name songs (vectorized)
        similarities[song_index] = -np.inf
        same_name_mask = (self.df["track_name"] == query_name).values
        same_name_mask[song_index] = False  # already excluded
        similarities[same_name_mask] = -np.inf

        # Over-fetch so post-dedup we still have top_k
        fetch = min(top_k * 3, len(similarities))
        ranked = np.argsort(similarities)[::-1][:fetch]
        top_sims = similarities[ranked]

        result = self.df.iloc[ranked][
            ["track_id", "track_name", "artists", "album_name", "track_genre", "popularity"]
        ].copy()
        result["similarity"] = top_sims
        result = result.reset_index(drop=True)
        return _dedupe_recs(result).head(top_k).reset_index(drop=True)

    def song_label(self, index: int) -> str:
        """Human-readable label for a song."""
        row = self.df.iloc[index]
        return f"{row['track_name']} - {row['artists']}"


# ── MMR (Maximum Marginal Relevance) reranking ────────────────────────────────


def rerank_mmr(
    engine: "RecommendationEngine",
    recs: pd.DataFrame,
    seed_index: int,
    lam: float = 0.7,
    top_k: int = 10,
) -> pd.DataFrame:
    """Rerank a recommendation set using Maximum Marginal Relevance (MMR).

    Reference: Carbonell & Goldstein (SIGIR '98).  At each step pick the
    candidate that maximizes:

        score(c) = lam * sim(c, query)  -  (1 - lam) * max_{s in S} sim(c, s)

    where S is the already-selected set.  lam=1 reduces to pure relevance;
    lam=0 reduces to "farthest-first" diversity.  The first pick is the
    most-relevant candidate (S is empty so the redundancy term is undefined).

    Operates on the engine's standardized feature matrix; cosine similarity
    is the similarity function throughout.
    """
    if recs.empty or top_k <= 0:
        return recs.head(0)

    # Resolve each rec back to its catalog row index.
    # Track recs_pos in parallel so we can index back into recs correctly even
    # when some rows are skipped (len(match)==0 guard).
    cand_indices: list[int] = []
    recs_pos: list[int] = []
    for i, (_, rec) in enumerate(recs.iterrows()):
        match = engine.df[
            (engine.df["track_name"] == rec["track_name"])
            & (engine.df["artists"] == rec["artists"])
        ]
        if len(match) == 0:
            continue
        cand_indices.append(int(match.index[0]))
        recs_pos.append(i)
    if not cand_indices:
        return recs.head(0)

    cand_idx_arr = np.asarray(cand_indices, dtype=int)
    cand_vecs = engine.feature_matrix[cand_idx_arr]
    query_vec = engine.feature_matrix[seed_index]

    # Cosine similarity to query (rel) and pairwise candidate-candidate (cc).
    cand_norms = np.linalg.norm(cand_vecs, axis=1)
    cand_norms = np.where(cand_norms > 0, cand_norms, 1.0)
    q_norm = np.linalg.norm(query_vec) or 1.0
    rel = (cand_vecs @ query_vec) / (cand_norms * q_norm)
    cc_sim = (cand_vecs @ cand_vecs.T) / np.outer(cand_norms, cand_norms)

    M = len(cand_idx_arr)
    selected: list[int] = []
    remaining: set[int] = set(range(M))
    mmr_scores: list[float] = []

    first = int(np.argmax(rel))
    selected.append(first)
    remaining.discard(first)
    mmr_scores.append(float(rel[first]))

    while len(selected) < top_k and remaining:
        rem = np.fromiter(remaining, dtype=int, count=len(remaining))
        redundancy = cc_sim[np.ix_(rem, selected)].max(axis=1)
        scores = lam * rel[rem] - (1.0 - lam) * redundancy
        local_best = int(np.argmax(scores))
        chosen = int(rem[local_best])
        selected.append(chosen)
        remaining.discard(chosen)
        mmr_scores.append(float(scores[local_best]))

    out = recs.iloc[[recs_pos[s] for s in selected]].copy().reset_index(drop=True)
    out["mmr_score"] = mmr_scores
    return out
