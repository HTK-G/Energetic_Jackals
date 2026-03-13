"""Interactive plotting utilities for Spotify analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.evaluation.metrics import cluster_distribution
from src.utils.constants import CONTINUOUS_AUDIO_FEATURES


def build_radar_chart(
    song_row: pd.Series,
    reference_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> go.Figure:
    """Create a normalized radar chart for one song's feature profile."""
    selected_features = feature_columns or [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "instrumentalness",
        "liveness",
        "speechiness",
        "tempo",
        "loudness",
        "popularity",
    ]
    selected_features = [feature for feature in selected_features if feature in reference_frame.columns and feature in song_row.index]

    normalized_values: list[float] = []
    for feature in selected_features:
        reference_series = pd.to_numeric(reference_frame[feature], errors="coerce")
        feature_min = float(reference_series.min())
        feature_max = float(reference_series.max())
        value = float(song_row[feature])

        if feature_max == feature_min:
            normalized_values.append(0.5)
        else:
            normalized_values.append((value - feature_min) / (feature_max - feature_min))

    closed_features = selected_features + selected_features[:1]
    closed_values = normalized_values + normalized_values[:1]

    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=closed_values,
            theta=closed_features,
            fill="toself",
            name=str(song_row.get("song_name", "Selected Song")),
        )
    )
    figure.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0.0, 1.0]}},
        showlegend=False,
        margin={"l": 16, "r": 16, "t": 40, "b": 16},
    )
    return figure


def build_embedding_scatter(
    catalog: pd.DataFrame,
    projection: np.ndarray,
    cluster_labels: np.ndarray | None = None,
    title: str = "Embedding Projection",
) -> go.Figure:
    """Scatter plot for 2D embeddings or projected embeddings."""
    plot_frame = catalog.copy()
    plot_frame["embedding_x"] = projection[:, 0]
    plot_frame["embedding_y"] = projection[:, 1]

    color_column = None
    if cluster_labels is not None:
        plot_frame["cluster"] = cluster_labels.astype(str)
        color_column = "cluster"
    elif "playlist_genre" in plot_frame.columns:
        color_column = "playlist_genre"

    figure = px.scatter(
        plot_frame,
        x="embedding_x",
        y="embedding_y",
        color=color_column,
        hover_name="song_name",
        hover_data={
            "artist_name": True,
            "playlist_genre": True,
            "embedding_x": False,
            "embedding_y": False,
        },
        title=title,
        opacity=0.78,
    )
    figure.update_layout(margin={"l": 16, "r": 16, "t": 60, "b": 16})
    return figure


def build_cluster_distribution_plot(labels: np.ndarray) -> go.Figure:
    """Bar chart showing the number of songs inside each cluster."""
    counts = cluster_distribution(labels).rename_axis("cluster").reset_index(name="song_count")
    counts["cluster"] = counts["cluster"].astype(str)

    figure = px.bar(
        counts,
        x="cluster",
        y="song_count",
        color="cluster",
        title="Cluster Distribution",
    )
    figure.update_layout(showlegend=False, margin={"l": 16, "r": 16, "t": 60, "b": 16})
    return figure


def build_correlation_heatmap(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> go.Figure:
    """Heatmap of pairwise feature correlations for continuous audio features."""
    selected_features = feature_columns or [feature for feature in CONTINUOUS_AUDIO_FEATURES if feature in frame.columns]
    correlation_frame = frame[selected_features].corr(numeric_only=True)

    figure = px.imshow(
        correlation_frame,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        title="Feature Correlation Heatmap",
    )
    figure.update_layout(margin={"l": 16, "r": 16, "t": 60, "b": 16})
    return figure
