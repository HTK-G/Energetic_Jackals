"""Feature engineering, encoding, and scaling for the Spotify dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "clean_dataset_final.csv"

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "key",
    "mode",
]

# After sine/cosine encoding of key, key and mode are replaced by key_sin, key_cos, mode
FEATURE_COLUMNS_ENCODED = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "key_sin",
    "key_cos",
    "mode",
]


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned Spotify dataset."""
    resolved = Path(path) if path else DATA_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset not found at {resolved}")
    return pd.read_csv(resolved)


def encode_key_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the integer `key` column (0-11) with sine/cosine encoding."""
    out = df.copy()
    radians = 2 * np.pi * out["key"] / 12
    out["key_sin"] = np.sin(radians)
    out["key_cos"] = np.cos(radians)
    out = out.drop(columns=["key"])
    return out


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """Build a standardized 12D feature matrix from the raw dataset.

    Returns:
        feature_matrix: (n_songs, 12) scaled numpy array
        scaler: fitted StandardScaler (for transforming new data)
        df_encoded: DataFrame with key_sin/key_cos columns added
    """
    # Select only the audio features we need
    for col in AUDIO_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df_encoded = encode_key_cyclical(df)

    raw_features = df_encoded[FEATURE_COLUMNS_ENCODED].values.astype(float)

    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(raw_features)

    return feature_matrix, scaler, df_encoded
