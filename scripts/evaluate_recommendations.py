"""Evaluate 6 recommendation methods on 2 metrics (Genre Hit Rate, Coverage).

Runs once after `precompute.py`:
    uv run python -m scripts.evaluate_recommendations

Output: `artifacts/recommendation_eval.joblib`

Methods compared (3 baselines + 3 of ours):

    random        — Uniform random K songs from the catalog.
    popularity    — Top-K most popular songs (same for every query — cold-start fallback).
    genre_match   — Random K songs sharing the query's track_genre.
    knn           — RecommendationEngine.recommend (full-feature cosine KNN).
    kmeans_cluster— RecommendationEngine.recommend_by_cluster (best K-Means model).
    gmm_posterior — RecommendationEngine.recommend_by_gmm (best GMM-full model).

Metrics:

    A. Genre Hit Rate.  Fraction of top-K recs that share at least one
       label with the query's `all_genres` set.  Range [0, 1] — higher is
       "recommendations look like the query."
    B. Genre Coverage.  Number of distinct `track_genre` values among the
       top-K recs.  Range [1, K] — higher is "recommendations span genres
       (good if you trust feature similarity over brittle genre labels)."

Sample size = 500 queries (fixed `random_state=42`); top_k = 10.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluate import parse_genre_set
from src.recommend import RecommendationEngine

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
SAMPLE_SIZE = 500
TOP_K = 10
RANDOM_STATE = 42


def _query_genres(df: pd.DataFrame, idx: int) -> set[str]:
    """Genre set for a query — falls back to track_genre if all_genres missing."""
    genres = parse_genre_set(df.iloc[idx].get("all_genres", ""))
    if not genres:
        g = df.iloc[idx].get("track_genre", "")
        if isinstance(g, str) and g.strip():
            genres = {g.strip().lower()}
    return genres


def _rec_genres(df: pd.DataFrame, indices: np.ndarray) -> list[set[str]]:
    """Genre sets for each recommended index (used for hit rate)."""
    out: list[set[str]] = []
    for idx in indices:
        g = parse_genre_set(df.iloc[idx].get("all_genres", ""))
        if not g:
            track_g = df.iloc[idx].get("track_genre", "")
            if isinstance(track_g, str) and track_g.strip():
                g = {track_g.strip().lower()}
        out.append(g)
    return out


def _hit_rate(query_genres: set[str], rec_genre_sets: list[set[str]]) -> float:
    if not rec_genre_sets:
        return 0.0
    if not query_genres:
        return 0.0
    hits = sum(1 for g in rec_genre_sets if query_genres & g)
    return hits / len(rec_genre_sets)


def _coverage(rec_track_genres: pd.Series) -> int:
    return int(rec_track_genres.nunique())


def _recs_to_indices(df: pd.DataFrame, recs: pd.DataFrame) -> np.ndarray:
    """Map a recommendation DataFrame back to catalog row indices."""
    out: list[int] = []
    for _, r in recs.iterrows():
        m = df[(df["track_name"] == r["track_name"]) & (df["artists"] == r["artists"])]
        if len(m):
            out.append(int(m.index[0]))
    return np.array(out, dtype=int)


# ── Baseline samplers ──────────────────────────────────────────────────────────


def baseline_random(df: pd.DataFrame, query_idx: int, top_k: int,
                    rng: np.random.Generator) -> np.ndarray:
    candidates = np.setdiff1d(np.arange(len(df)), [query_idx])
    return rng.choice(candidates, size=min(top_k, len(candidates)), replace=False)


def baseline_popularity(df: pd.DataFrame, query_idx: int, top_k: int,
                        pop_top_indices: np.ndarray) -> np.ndarray:
    """Always returns the global top-K by popularity, sans the query itself."""
    return pop_top_indices[pop_top_indices != query_idx][:top_k]


def baseline_genre_match(df: pd.DataFrame, query_idx: int, top_k: int,
                         rng: np.random.Generator) -> np.ndarray:
    g = df.iloc[query_idx].get("track_genre", "")
    pool = np.flatnonzero((df["track_genre"] == g).values)
    pool = pool[pool != query_idx]
    if len(pool) == 0:
        return baseline_random(df, query_idx, top_k, rng)
    return rng.choice(pool, size=min(top_k, len(pool)), replace=False)


# ── Main loop ──────────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading artifacts…")
    feats = joblib.load(ARTIFACTS_DIR / "feature_matrix.joblib")
    df = feats["df_encoded"]
    feature_matrix = feats["feature_matrix"]
    engine = RecommendationEngine(df, feature_matrix)
    km = joblib.load(ARTIFACTS_DIR / "kmeans_best.joblib")
    gmm = joblib.load(ARTIFACTS_DIR / "gmm_full_best.joblib")

    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = rng.choice(len(df), size=SAMPLE_SIZE, replace=False)
    pop_top_indices = df["popularity"].sort_values(ascending=False).index.values[: TOP_K + 5]

    print(f"Evaluating {SAMPLE_SIZE} queries × 6 methods × top_{TOP_K}…")
    raw: dict[str, dict[str, list[float]]] = {
        m: {"hit_rates": [], "coverages": []}
        for m in ["random", "popularity", "genre_match", "knn", "kmeans_cluster", "gmm_posterior"]
    }

    t0 = time.time()
    for i, qi in enumerate(sample_indices):
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{SAMPLE_SIZE}  ({time.time() - t0:.1f}s)")

        qi = int(qi)
        q_genres = _query_genres(df, qi)

        # Baselines (return integer index arrays directly)
        baselines = {
            "random": baseline_random(df, qi, TOP_K, rng),
            "popularity": baseline_popularity(df, qi, TOP_K, pop_top_indices),
            "genre_match": baseline_genre_match(df, qi, TOP_K, rng),
        }
        for name, idx_arr in baselines.items():
            rec_genre_sets = _rec_genres(df, idx_arr)
            raw[name]["hit_rates"].append(_hit_rate(q_genres, rec_genre_sets))
            raw[name]["coverages"].append(_coverage(df.iloc[idx_arr]["track_genre"]))

        # Ours (return DataFrames)
        ours = {
            "knn": engine.recommend(qi, top_k=TOP_K),
            "kmeans_cluster": engine.recommend_by_cluster(qi, km.labels, top_k=TOP_K),
            "gmm_posterior": engine.recommend_by_gmm(qi, gmm.probabilities, top_k=TOP_K),
        }
        for name, recs in ours.items():
            idx_arr = _recs_to_indices(df, recs)
            rec_genre_sets = _rec_genres(df, idx_arr)
            raw[name]["hit_rates"].append(_hit_rate(q_genres, rec_genre_sets))
            raw[name]["coverages"].append(_coverage(recs["track_genre"]) if len(recs) else 0)

    # Aggregate
    results = {}
    for name, vals in raw.items():
        hr = np.array(vals["hit_rates"], dtype=float)
        cov = np.array(vals["coverages"], dtype=float)
        results[name] = {
            "hit_rate_mean": float(hr.mean()),
            "hit_rate_std": float(hr.std()),
            "coverage_mean": float(cov.mean()),
            "coverage_std": float(cov.std()),
        }

    print(f"\nDone in {time.time() - t0:.1f}s.\n")
    print(f"{'Method':<18} {'HitRate (μ ± σ)':<20} {'Coverage (μ ± σ)':<20}")
    print("-" * 60)
    for name in ["random", "popularity", "genre_match", "knn", "kmeans_cluster", "gmm_posterior"]:
        r = results[name]
        print(f"{name:<18} {r['hit_rate_mean']:.3f} ± {r['hit_rate_std']:.3f}     "
              f"{r['coverage_mean']:.2f} ± {r['coverage_std']:.2f}")

    out = {
        "n_queries": SAMPLE_SIZE,
        "top_k": TOP_K,
        "random_state": RANDOM_STATE,
        "results": results,
        "raw_scores": raw,
    }
    out_path = ARTIFACTS_DIR / "recommendation_eval.joblib"
    joblib.dump(out, out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
