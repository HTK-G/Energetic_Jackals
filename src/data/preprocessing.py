"""Preprocessing pipeline for Spotify audio features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.constants import CATEGORICAL_AUDIO_FEATURES, CONTINUOUS_AUDIO_FEATURES


@dataclass(slots=True)
class PreparedData:
    """Container for cleaned tabular data and the derived feature matrix."""

    cleaned_frame: pd.DataFrame
    feature_matrix: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer
    continuous_features: list[str]
    categorical_features: list[str]


def _resolve_continuous_features(frame: pd.DataFrame) -> list[str]:
    return [feature for feature in CONTINUOUS_AUDIO_FEATURES if feature in frame.columns]


def _resolve_categorical_features(frame: pd.DataFrame) -> list[str]:
    return [feature for feature in CATEGORICAL_AUDIO_FEATURES if feature in frame.columns]


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Build a reusable preprocessing pipeline for numeric and categorical features."""
    continuous_features = _resolve_continuous_features(frame)
    categorical_features = _resolve_categorical_features(frame)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("continuous", numeric_pipeline, continuous_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def _clean_numeric_columns(frame: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    cleaned = frame.copy()

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if numeric_columns:
        medians = cleaned[numeric_columns].median(numeric_only=True)
        cleaned[numeric_columns] = cleaned[numeric_columns].fillna(medians)

    return cleaned


def _clean_categorical_columns(frame: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in categorical_columns:
        mode_series = cleaned[column].mode(dropna=True)
        fallback = int(mode_series.iloc[0]) if not mode_series.empty else 0
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(fallback).astype(int)
    return cleaned


def preprocess_spotify_dataframe(frame: pd.DataFrame) -> PreparedData:
    """Clean the Spotify dataset and return a feature matrix ready for modeling."""
    working_frame = frame.copy()
    continuous_features = _resolve_continuous_features(working_frame)
    categorical_features = _resolve_categorical_features(working_frame)

    for metadata_column, fallback_value in {
        "song_name": "Unknown Song",
        "artist_name": "Unknown Artist",
        "album_name": "Unknown Album",
        "playlist_genre": "unknown",
    }.items():
        if metadata_column in working_frame.columns:
            working_frame[metadata_column] = working_frame[metadata_column].fillna(fallback_value).astype(str)

    working_frame = _clean_numeric_columns(working_frame, continuous_features)
    working_frame = _clean_categorical_columns(working_frame, categorical_features)

    preprocessor = build_preprocessor(working_frame)
    feature_matrix = preprocessor.fit_transform(working_frame)
    feature_names = preprocessor.get_feature_names_out().tolist()

    return PreparedData(
        cleaned_frame=working_frame.reset_index(drop=True),
        feature_matrix=np.asarray(feature_matrix, dtype=float),
        feature_names=feature_names,
        preprocessor=preprocessor,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )
