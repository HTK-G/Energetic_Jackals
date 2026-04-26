"""Derive data-driven start/end feature vectors for Music Journey scenarios.

Run once after `precompute.py`:
    uv run python -m scripts.derive_scenario_mappings

Output: `artifacts/scenario_mappings.joblib`

For each scenario, find songs whose `track_genre` contains any of the scenario's
genre keywords (case-insensitive substring), then compute per-feature percentiles
over those songs:

    start    = 25th percentile      (low end of the source genres' range)
    centroid = 50th percentile      (median, useful for radio mode)
    end      = 75th percentile      (high end of the source genres' range)

The 4 trajectory features all live in [0, 1]:
    energy, valence, danceability   — already in [0, 1] in the raw dataset
    tempo_norm                      — clip tempo to [50, 200] BPM then (tempo - 50) / 150

This is "semi data-driven": the scenario -> keyword mapping is hand-picked,
but the resulting feature vectors come from real percentiles over matched songs,
not from typed-in defaults.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

TRAJECTORY_FEATURES = ["energy", "valence", "danceability", "tempo_norm"]

SCENARIO_KEYWORDS: dict[str, list[str]] = {
    "workout":     ["workout", "electro", "dance", "hip-hop"],
    "focus":       ["classical", "piano", "study", "ambient"],
    "wind down":   ["sleep", "chill", "acoustic", "singer-songwriter"],
    "party":       ["party", "dance", "disco", "pop"],
    "commute":     ["pop", "indie-pop", "rock", "alternative"],
    "rainy night": ["jazz", "blues", "sad", "r-n-b"],
}


def normalize_tempo(tempo: pd.Series) -> pd.Series:
    """Clip tempo to [50, 200] BPM then scale to [0, 1] via (tempo - 50) / 150."""
    return (tempo.clip(lower=50, upper=200) - 50) / 150


def trajectory_dataframe(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """Project the catalog onto the 4-D trajectory feature space."""
    return pd.DataFrame({
        "energy": df_encoded["energy"].astype(float),
        "valence": df_encoded["valence"].astype(float),
        "danceability": df_encoded["danceability"].astype(float),
        "tempo_norm": normalize_tempo(df_encoded["tempo"].astype(float)),
    })


def derive_one_scenario(
    df_encoded: pd.DataFrame,
    traj_df: pd.DataFrame,
    keywords: list[str],
) -> dict:
    """Find songs matching any keyword, return start/centroid/end percentiles."""
    genre_lower = df_encoded["track_genre"].fillna("").str.lower()
    mask = pd.Series(False, index=df_encoded.index)
    matched_genres: set[str] = set()
    for kw in keywords:
        kw_mask = genre_lower.str.contains(kw.lower(), regex=False)
        mask = mask | kw_mask
        matched_genres.update(df_encoded.loc[kw_mask, "track_genre"].unique())

    subset = traj_df[mask]
    if len(subset) == 0:
        raise ValueError(f"No songs matched keywords {keywords}")

    percentiles = subset.quantile([0.25, 0.5, 0.75])
    return {
        "start": percentiles.loc[0.25].values.astype(float),
        "centroid": percentiles.loc[0.5].values.astype(float),
        "end": percentiles.loc[0.75].values.astype(float),
        "source_genres": list(keywords),
        "matched_genres": sorted(matched_genres),
        "n_songs": int(len(subset)),
    }


def main() -> None:
    feats = joblib.load(ARTIFACTS_DIR / "feature_matrix.joblib")
    df_encoded = feats["df_encoded"]
    traj_df = trajectory_dataframe(df_encoded)

    print(f"Catalog: {len(df_encoded):,} songs across {df_encoded['track_genre'].nunique()} genres")
    print(f"Trajectory features: {TRAJECTORY_FEATURES}\n")

    mappings: dict[str, dict] = {}
    for scenario, keywords in SCENARIO_KEYWORDS.items():
        t0 = time.time()
        result = derive_one_scenario(df_encoded, traj_df, keywords)
        mappings[scenario] = result
        miss = [
            kw for kw in keywords
            if not any(kw.lower() in g.lower() for g in result["matched_genres"])
        ]
        if miss:
            print(f"  ! {scenario}: keywords with no genre match: {miss}")
        print(
            f"  {scenario:12s} n={result['n_songs']:>6d}  "
            f"start={np.round(result['start'], 2).tolist()}  "
            f"end={np.round(result['end'], 2).tolist()}  "
            f"({time.time() - t0:.2f}s)"
        )

    out_path = ARTIFACTS_DIR / "scenario_mappings.joblib"
    joblib.dump(mappings, out_path)
    print(f"\nSaved {len(mappings)} mappings -> {out_path}")


if __name__ == "__main__":
    main()
