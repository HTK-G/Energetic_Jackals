"""Dataset loading and schema normalization utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.constants import RAW_DATA_DIR
from src.utils.demo_data import generate_demo_spotify_dataset

SCHEMA_ALIASES = {
    "track_id": ["track_id", "id", "spotify_id", "uri"],
    "song_name": ["song_name", "track_name", "name", "title"],
    "artist_name": ["artist_name", "artists", "artist", "artist_names"],
    "album_name": ["album_name", "album", "album_title"],
    "playlist_genre": ["playlist_genre", "genre", "track_genre"],
    "danceability": ["danceability"],
    "energy": ["energy"],
    "loudness": ["loudness"],
    "speechiness": ["speechiness"],
    "acousticness": ["acousticness"],
    "instrumentalness": ["instrumentalness"],
    "liveness": ["liveness"],
    "valence": ["valence"],
    "tempo": ["tempo"],
    "duration_ms": ["duration_ms", "duration"],
    "popularity": ["popularity"],
    "key": ["key"],
    "mode": ["mode"],
    "time_signature": ["time_signature", "timesignature", "time signature"],
}


def _normalize_column_name(column_name: str) -> str:
    return column_name.strip().lower().replace(" ", "_").replace("-", "_")


def standardize_spotify_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize source columns into a stable internal schema."""
    renamed_frame = frame.copy()
    normalized_lookup = {_normalize_column_name(column): column for column in renamed_frame.columns}
    rename_map: dict[str, str] = {}

    for target_name, aliases in SCHEMA_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_column_name(alias)
            if normalized_alias in normalized_lookup:
                rename_map[normalized_lookup[normalized_alias]] = target_name
                break

    renamed_frame = renamed_frame.rename(columns=rename_map)

    if "song_name" not in renamed_frame.columns:
        renamed_frame["song_name"] = [f"Song {index + 1}" for index in range(len(renamed_frame))]
    if "artist_name" not in renamed_frame.columns:
        renamed_frame["artist_name"] = "Unknown Artist"
    if "album_name" not in renamed_frame.columns:
        renamed_frame["album_name"] = "Unknown Album"
    if "playlist_genre" not in renamed_frame.columns:
        renamed_frame["playlist_genre"] = "unknown"
    if "track_id" not in renamed_frame.columns:
        renamed_frame["track_id"] = renamed_frame.index.map(lambda value: f"track-{value}")

    renamed_frame["track_id"] = renamed_frame["track_id"].astype(str)
    renamed_frame["song_name"] = renamed_frame["song_name"].astype(str)
    renamed_frame["artist_name"] = renamed_frame["artist_name"].astype(str)
    renamed_frame["album_name"] = renamed_frame["album_name"].astype(str)
    renamed_frame["playlist_genre"] = renamed_frame["playlist_genre"].astype(str)
    return renamed_frame


def detect_dataset_path(search_dir: Path = RAW_DATA_DIR) -> Path | None:
    """Find the first CSV dataset under the raw data directory."""
    csv_files = sorted(search_dir.glob("*.csv"))
    return csv_files[0] if csv_files else None


def load_spotify_dataset(dataset_path: str | Path | None = None, sample_size: int | None = None) -> pd.DataFrame:
    """Load the Spotify dataset or fall back to a generated demo catalog."""
    resolved_path = Path(dataset_path).expanduser() if dataset_path else detect_dataset_path()

    if resolved_path is None or not resolved_path.exists():
        frame = generate_demo_spotify_dataset()
    else:
        frame = pd.read_csv(resolved_path)

    standardized_frame = standardize_spotify_schema(frame)

    if sample_size is not None and sample_size > 0 and len(standardized_frame) > sample_size:
        standardized_frame = standardized_frame.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        standardized_frame = standardized_frame.reset_index(drop=True)

    return standardized_frame
