"""Deterministic demo dataset generation for local development."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.constants import DEMO_RANDOM_SEED


def generate_demo_spotify_dataset(num_songs: int = 500, seed: int = DEMO_RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic Spotify-like dataset with cluster structure."""
    rng = np.random.default_rng(seed)

    moods = [
        {
            "label": "Night Drive",
            "genre": "electronic",
            "center": {
                "danceability": 0.82,
                "energy": 0.88,
                "loudness": -5.5,
                "speechiness": 0.06,
                "acousticness": 0.08,
                "instrumentalness": 0.18,
                "liveness": 0.14,
                "valence": 0.62,
                "tempo": 124.0,
                "duration_ms": 210000.0,
                "popularity": 72.0,
                "key": 9,
                "mode": 1,
                "time_signature": 4,
            },
        },
        {
            "label": "Coffeehouse",
            "genre": "acoustic",
            "center": {
                "danceability": 0.48,
                "energy": 0.34,
                "loudness": -12.5,
                "speechiness": 0.05,
                "acousticness": 0.88,
                "instrumentalness": 0.07,
                "liveness": 0.11,
                "valence": 0.43,
                "tempo": 92.0,
                "duration_ms": 238000.0,
                "popularity": 58.0,
                "key": 5,
                "mode": 1,
                "time_signature": 4,
            },
        },
        {
            "label": "Festival",
            "genre": "dance",
            "center": {
                "danceability": 0.76,
                "energy": 0.93,
                "loudness": -4.3,
                "speechiness": 0.09,
                "acousticness": 0.05,
                "instrumentalness": 0.03,
                "liveness": 0.27,
                "valence": 0.74,
                "tempo": 130.0,
                "duration_ms": 202000.0,
                "popularity": 79.0,
                "key": 2,
                "mode": 1,
                "time_signature": 4,
            },
        },
        {
            "label": "Deep Focus",
            "genre": "ambient",
            "center": {
                "danceability": 0.28,
                "energy": 0.22,
                "loudness": -16.5,
                "speechiness": 0.04,
                "acousticness": 0.74,
                "instrumentalness": 0.91,
                "liveness": 0.08,
                "valence": 0.25,
                "tempo": 78.0,
                "duration_ms": 265000.0,
                "popularity": 46.0,
                "key": 0,
                "mode": 0,
                "time_signature": 4,
            },
        },
    ]

    rows: list[dict[str, object]] = []
    noise_scale = {
        "danceability": 0.06,
        "energy": 0.07,
        "loudness": 1.2,
        "speechiness": 0.03,
        "acousticness": 0.08,
        "instrumentalness": 0.1,
        "liveness": 0.05,
        "valence": 0.08,
        "tempo": 8.0,
        "duration_ms": 16000.0,
        "popularity": 9.0,
    }

    for index in range(num_songs):
        mood = moods[index % len(moods)]
        center = mood["center"]
        row = {
            "track_id": f"demo-{index:05d}",
            "song_name": f"{mood['label']} Track {index + 1}",
            "artist_name": f"Synthetic Artist {index % 37}",
            "album_name": f"{mood['label']} Sessions",
            "playlist_genre": mood["genre"],
        }

        for feature, feature_center in center.items():
            if feature in {"key", "mode", "time_signature"}:
                row[feature] = feature_center
                continue

            sampled_value = float(rng.normal(feature_center, noise_scale[feature]))
            if feature in {"danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"}:
                sampled_value = float(np.clip(sampled_value, 0.0, 1.0))
            if feature == "popularity":
                sampled_value = float(np.clip(sampled_value, 0.0, 100.0))
            if feature == "duration_ms":
                sampled_value = float(max(sampled_value, 60000.0))
            row[feature] = sampled_value

        row["explicit"] = int(rng.random() > 0.82)
        row["release_year"] = int(rng.integers(1998, 2025))
        rows.append(row)

    return pd.DataFrame(rows)
