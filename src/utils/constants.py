"""Project-wide constants for Spotify feature engineering."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEMO_RANDOM_SEED = 42

CONTINUOUS_AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "popularity",
]

CATEGORICAL_AUDIO_FEATURES = ["key", "mode", "time_signature"]

DEFAULT_FEATURE_COLUMNS = CONTINUOUS_AUDIO_FEATURES + CATEGORICAL_AUDIO_FEATURES

CORE_METADATA_COLUMNS = [
    "track_id",
    "song_name",
    "artist_name",
    "album_name",
    "playlist_genre",
]
