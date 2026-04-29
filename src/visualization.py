"""Russell emotion-space visualizations for Music Journey.

Main projection: energy (x-axis) × valence (y-axis), the standard
Russell circumplex model of affect.  Danceability and tempo are encoded
as marker size and color saturation respectively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Russell quadrant annotations
_QUADRANT_ANNOTATIONS = [
    dict(x=0.85, y=0.85, text="😄 Excited / Happy", showarrow=False,
         font=dict(size=11, color="rgba(255,255,255,0.5)")),
    dict(x=0.15, y=0.85, text="😠 Angry / Tense", showarrow=False,
         font=dict(size=11, color="rgba(255,255,255,0.5)")),
    dict(x=0.15, y=0.15, text="😢 Sad / Depressed", showarrow=False,
         font=dict(size=11, color="rgba(255,255,255,0.5)")),
    dict(x=0.85, y=0.15, text="😌 Calm / Peaceful", showarrow=False,
         font=dict(size=11, color="rgba(255,255,255,0.5)")),
]

_PLOT_BG = "rgba(0,0,0,0)"
_PAPER_BG = "rgba(0,0,0,0)"
_GRID_COLOR = "rgba(255,255,255,0.08)"
_ACCENT = "#c8956c"


def plot_journey(
    waypoints: list[np.ndarray],
    selected_songs: pd.DataFrame,
    background_df: pd.DataFrame,
    start: np.ndarray,
    end: np.ndarray,
) -> go.Figure:
    """Russell emotion-space 2D visualization of a trajectory playlist.

    Layers:
        1. Background scatter (sampled, low opacity)
        2. Trajectory line + waypoints (dashed, hollow circles)
        3. Selected songs (filled circles, size ∝ danceability)
        4. Start (★) + End (■) markers

    Parameters
    ----------
    waypoints : list of 4-D arrays (only first 2 dims used: energy, valence)
    selected_songs : DataFrame with actual_energy, actual_valence, actual_danceability,
                     track_name, artists columns
    background_df : DataFrame with energy, valence columns (sampled subset)
    start, end : 4-D arrays
    """
    fig = go.Figure()

    # Layer 1: Background
    bg_sample = background_df.sample(n=min(2000, len(background_df)), random_state=42)
    fig.add_trace(go.Scatter(
        x=bg_sample["energy"], y=bg_sample["valence"],
        mode="markers",
        marker=dict(size=3, color="rgba(200,200,200,0.15)"),
        name="All songs",
        hoverinfo="skip",
        showlegend=False,
    ))

    # Layer 2: Trajectory line + waypoints
    wp_e = [w[0] for w in waypoints]
    wp_v = [w[1] for w in waypoints]
    fig.add_trace(go.Scatter(
        x=wp_e, y=wp_v,
        mode="lines+markers",
        line=dict(dash="dash", color="rgba(255,255,255,0.4)", width=2),
        marker=dict(size=8, symbol="circle-open", color="rgba(255,255,255,0.5)",
                    line=dict(width=1, color="rgba(255,255,255,0.5)")),
        name="Target waypoints",
        hoverinfo="skip",
    ))

    # Layer 3: Selected songs
    dance = selected_songs["actual_danceability"].values
    sizes = 8 + 20 * dance  # range [8, 28]
    fig.add_trace(go.Scatter(
        x=selected_songs["actual_energy"],
        y=selected_songs["actual_valence"],
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=_ACCENT,
            opacity=0.85,
            line=dict(width=1, color="white"),
        ),
        text=[str(i + 1) for i in range(len(selected_songs))],
        textposition="top center",
        textfont=dict(size=9, color="white"),
        customdata=np.column_stack([
            selected_songs["track_name"].values,
            selected_songs["artists"].values,
        ]),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                      "Energy: %{x:.2f}<br>Valence: %{y:.2f}<extra></extra>",
        name="Selected songs",
    ))

    # Layer 4: Start + End markers
    fig.add_trace(go.Scatter(
        x=[start[0]], y=[start[1]],
        mode="markers",
        marker=dict(size=16, symbol="star", color="#4ade80",
                    line=dict(width=1, color="white")),
        name="Start",
    ))
    fig.add_trace(go.Scatter(
        x=[end[0]], y=[end[1]],
        mode="markers",
        marker=dict(size=14, symbol="square", color="#f87171",
                    line=dict(width=1, color="white")),
        name="End",
    ))

    fig.update_layout(
        xaxis=dict(title="Energy →", range=[-0.05, 1.05],
                   gridcolor=_GRID_COLOR, zeroline=False),
        yaxis=dict(title="Valence →", range=[-0.05, 1.05],
                   gridcolor=_GRID_COLOR, zeroline=False),
        plot_bgcolor=_PLOT_BG,
        paper_bgcolor=_PAPER_BG,
        font=dict(color="white"),
        annotations=_QUADRANT_ANNOTATIONS,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10)),
        margin=dict(l=40, r=20, t=40, b=40),
        height=500,
    )
    return fig


def plot_endless_history(
    history_df: pd.DataFrame,
    background_df: pd.DataFrame,
) -> go.Figure:
    """Connected scatter of played songs in Russell space showing drift path.

    Parameters
    ----------
    history_df : DataFrame with actual_energy, actual_valence, track_name, artists
    background_df : DataFrame with energy, valence columns
    """
    fig = go.Figure()

    # Background
    bg_sample = background_df.sample(n=min(2000, len(background_df)), random_state=42)
    fig.add_trace(go.Scatter(
        x=bg_sample["energy"], y=bg_sample["valence"],
        mode="markers",
        marker=dict(size=3, color="rgba(200,200,200,0.15)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    if len(history_df) > 0:
        # Path line
        fig.add_trace(go.Scatter(
            x=history_df["actual_energy"],
            y=history_df["actual_valence"],
            mode="lines+markers",
            line=dict(color=_ACCENT, width=2),
            marker=dict(
                size=10, color=_ACCENT, opacity=0.9,
                line=dict(width=1, color="white"),
            ),
            customdata=np.column_stack([
                history_df["track_name"].values,
                history_df["artists"].values,
            ]),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                          "Energy: %{x:.2f}<br>Valence: %{y:.2f}<extra></extra>",
            name="Played songs",
        ))

        # Highlight first and last
        fig.add_trace(go.Scatter(
            x=[history_df.iloc[0]["actual_energy"]],
            y=[history_df.iloc[0]["actual_valence"]],
            mode="markers",
            marker=dict(size=14, symbol="star", color="#4ade80",
                        line=dict(width=1, color="white")),
            name="First song",
        ))

    fig.update_layout(
        xaxis=dict(title="Energy →", range=[-0.05, 1.05],
                   gridcolor=_GRID_COLOR, zeroline=False),
        yaxis=dict(title="Valence →", range=[-0.05, 1.05],
                   gridcolor=_GRID_COLOR, zeroline=False),
        plot_bgcolor=_PLOT_BG,
        paper_bgcolor=_PAPER_BG,
        font=dict(color="white"),
        annotations=_QUADRANT_ANNOTATIONS,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10)),
        margin=dict(l=40, r=20, t=40, b=40),
        height=500,
    )
    return fig
