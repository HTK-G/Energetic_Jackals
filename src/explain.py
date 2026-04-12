"""Feature-level explanation and radar charts for recommendation results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.features import FEATURE_COLUMNS_ENCODED

# Display-friendly names for radar chart axes
# key_sin and key_cos are the sine/cosine encoding of musical key (0-11 pitch class).
# This encoding preserves cyclical distance: key 0 (C) and key 11 (B) are neighbors.
DISPLAY_NAMES = {
    "danceability": "Danceability",
    "energy": "Energy",
    "loudness": "Loudness",
    "speechiness": "Speechiness",
    "acousticness": "Acousticness",
    "instrumentalness": "Instrumentalness",
    "liveness": "Liveness",
    "valence": "Valence",
    "tempo": "Tempo",
    "key_sin": "Key (Pitch Angle)",
    "key_cos": "Key (Pitch Depth)",
    "mode": "Mode (Major/Minor)",
}

QUERY_COLOR = "rgba(31, 119, 180, 0.5)"     # blue
QUERY_LINE_COLOR = "rgba(31, 119, 180, 1)"
REC_COLOR = "rgba(255, 127, 14, 0.5)"       # orange
REC_LINE_COLOR = "rgba(255, 127, 14, 1)"
SINGLE_COLOR = "rgba(99, 110, 250, 0.5)"    # indigo
SINGLE_LINE_COLOR = "rgba(99, 110, 250, 1)"


def _display_name(feature: str) -> str:
    return DISPLAY_NAMES.get(feature, feature)


def feature_difference(query_row: pd.Series, rec_row: pd.Series) -> pd.DataFrame:
    """Compute per-feature absolute difference between query and recommended song."""
    rows = []
    for feat in FEATURE_COLUMNS_ENCODED:
        q_val = float(query_row[feat])
        r_val = float(rec_row[feat])
        rows.append({
            "feature": _display_name(feat),
            "query_value": q_val,
            "recommended_value": r_val,
            "abs_difference": abs(q_val - r_val),
        })
    return pd.DataFrame(rows).sort_values("abs_difference")


def explain_recommendation(query_row: pd.Series, rec_row: pd.Series, top_n: int = 3) -> str:
    """Generate a natural-language explanation for why a song was recommended."""
    diff = feature_difference(query_row, rec_row)
    most_similar = diff.head(top_n)
    most_different = diff.tail(top_n).iloc[::-1]

    similar_parts = []
    for _, row in most_similar.iterrows():
        similar_parts.append(f"{row['feature']} ({row['query_value']:.2f} vs {row['recommended_value']:.2f})")

    different_parts = []
    for _, row in most_different.iterrows():
        different_parts.append(f"{row['feature']} ({row['query_value']:.2f} vs {row['recommended_value']:.2f})")

    return (
        f"Similar in: {', '.join(similar_parts)}. "
        f"Differs in: {', '.join(different_parts)}."
    )


def _normalize_for_radar(reference_df: pd.DataFrame, value: float, col: str) -> float:
    col_min = float(reference_df[col].min())
    col_max = float(reference_df[col].max())
    if col_max == col_min:
        return 0.5
    return (value - col_min) / (col_max - col_min)


def build_comparison_radar(
    query_row: pd.Series,
    rec_row: pd.Series,
    reference_df: pd.DataFrame,
    query_name: str = "Query",
    rec_name: str = "Recommended",
) -> go.Figure:
    """Radar chart comparing query song vs recommended song (normalized 0-1)."""
    features = [f for f in FEATURE_COLUMNS_ENCODED if f in reference_df.columns]
    display_labels = [_display_name(f) for f in features]

    q_vals = [_normalize_for_radar(reference_df, float(query_row[f]), f) for f in features]
    r_vals = [_normalize_for_radar(reference_df, float(rec_row[f]), f) for f in features]

    # Close the polygon
    labels_closed = display_labels + [display_labels[0]]
    q_closed = q_vals + [q_vals[0]]
    r_closed = r_vals + [r_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=q_closed, theta=labels_closed, fill="toself",
        name=query_name,
        fillcolor=QUERY_COLOR,
        line={"color": QUERY_LINE_COLOR, "width": 2},
    ))
    fig.add_trace(go.Scatterpolar(
        r=r_closed, theta=labels_closed, fill="toself",
        name=rec_name,
        fillcolor=REC_COLOR,
        line={"color": REC_LINE_COLOR, "width": 2},
    ))
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=True,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )
    return fig


def build_single_radar(
    song_row: pd.Series,
    reference_df: pd.DataFrame,
    song_name: str = "Song",
) -> go.Figure:
    """Radar chart for a single song's feature profile."""
    features = [f for f in FEATURE_COLUMNS_ENCODED if f in reference_df.columns]
    display_labels = [_display_name(f) for f in features]

    vals = [_normalize_for_radar(reference_df, float(song_row[f]), f) for f in features]
    labels_closed = display_labels + [display_labels[0]]
    vals_closed = vals + [vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=labels_closed, fill="toself", name=song_name,
        fillcolor=SINGLE_COLOR,
        line={"color": SINGLE_LINE_COLOR, "width": 2},
    ))
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=False,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )
    return fig
