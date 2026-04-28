"""Phase 2: Cluster visualization, profiling, and evaluation page.

All training is precomputed offline by `scripts/precompute.py`. This page only
loads pickled artifacts.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.clustering import ClusterResult, TuningResult
from src.evaluate import ClusterMetrics, metrics_comparison_table
from src.explain import DISPLAY_NAMES
from src.features import FEATURE_COLUMNS_ENCODED


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
REQUIRED_ARTIFACTS = [
    "feature_matrix.joblib",
    "pca_2d.joblib",
    "tuning_kmeans.joblib",
    "tuning_gmm_full.joblib",
    "tuning_gmm_diag.joblib",
    "kmeans_best.joblib",
    "gmm_full_best.joblib",
    "gmm_diag_best.joblib",
    "metrics_comparison.joblib",
]


@st.cache_resource
def _load_artifacts() -> dict:
    missing = [name for name in REQUIRED_ARTIFACTS if not (ARTIFACTS_DIR / name).exists()]
    if missing:
        st.error(
            "Precomputed artifacts not found: "
            + ", ".join(missing)
            + "\n\nRun `uv run python -m scripts.precompute` once to generate them."
        )
        st.stop()
    return {name.removesuffix(".joblib"): joblib.load(ARTIFACTS_DIR / name) for name in REQUIRED_ARTIFACTS}


# ── Cluster profiling helpers ────────────────────────────────────────────────


def _cluster_feature_means(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df_work = df.copy()
    df_work["cluster"] = labels
    feature_cols = [c for c in FEATURE_COLUMNS_ENCODED if c in df_work.columns]
    return df_work.groupby("cluster")[feature_cols].mean()


def _cluster_top_genres(df: pd.DataFrame, labels: np.ndarray, top_n: int = 5) -> dict[int, list[tuple[str, int]]]:
    df_work = df.copy()
    df_work["cluster"] = labels
    result = {}
    for cid in sorted(df_work["cluster"].unique()):
        genre_counts = df_work[df_work["cluster"] == cid]["track_genre"].value_counts().head(top_n)
        result[int(cid)] = list(zip(genre_counts.index.tolist(), genre_counts.values.tolist()))
    return result


def _auto_label_cluster(mean_row: pd.Series) -> str:
    labels = []

    if mean_row.get("energy", 0) > 0.7 and mean_row.get("danceability", 0) > 0.7:
        labels.append("High-energy dance")
    elif mean_row.get("energy", 0) > 0.7:
        labels.append("High-energy")
    elif mean_row.get("energy", 0) < 0.3:
        labels.append("Low-energy")

    if mean_row.get("acousticness", 0) > 0.6:
        labels.append("Acoustic")
    if mean_row.get("instrumentalness", 0) > 0.5:
        labels.append("Instrumental")
    if mean_row.get("speechiness", 0) > 0.3:
        labels.append("Spoken/Speech-heavy")
    if mean_row.get("valence", 0) > 0.7:
        labels.append("Upbeat/Positive")
    elif mean_row.get("valence", 0) < 0.3:
        labels.append("Dark/Melancholic")

    if mean_row.get("tempo", 0) > 140:
        labels.append("Fast tempo")
    elif mean_row.get("tempo", 0) < 90:
        labels.append("Slow tempo")

    return ", ".join(labels) if labels else "Mixed"


# ── Visualization helpers ────────────────────────────────────────────────────


def _scatter_plot(
    projection: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    title: str,
    method_label: str,
) -> go.Figure:
    plot_df = pd.DataFrame({
        f"{method_label}_1": projection[:, 0],
        f"{method_label}_2": projection[:, 1],
        "cluster": labels.astype(str),
        "track_name": df["track_name"].values,
        "artists": df["artists"].values,
        "genre": df["track_genre"].values,
    })
    fig = px.scatter(
        plot_df,
        x=f"{method_label}_1",
        y=f"{method_label}_2",
        color="cluster",
        hover_data=["track_name", "artists", "genre"],
        title=title,
        opacity=0.6,
    )
    fig.update_layout(margin={"l": 20, "r": 20, "t": 50, "b": 20})
    return fig


def _tuning_elbow_chart(tuning: TuningResult, metric_name: str, values: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tuning.k_range, y=values, mode="lines+markers", name=metric_name,
    ))
    fig.update_layout(
        title=f"{metric_name} vs K",
        xaxis_title="K (number of clusters)",
        yaxis_title=metric_name,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


# ── Page layout ──────────────────────────────────────────────────────────────


artifacts = _load_artifacts()
df_encoded: pd.DataFrame = artifacts["feature_matrix"]["df_encoded"]
feature_matrix: np.ndarray = artifacts["feature_matrix"]["feature_matrix"]
pca_2d: np.ndarray = artifacts["pca_2d"]

ALGO_OPTIONS = {
    "K-Means (CustomKMeans)": {
        "result_key": "kmeans_best",
        "tuning_key": "tuning_kmeans",
        "tuning_secondary": ("Inertia (Elbow)", "inertias"),
        "tuning_label": "silhouette",
    },
    "GMM (full covariance)": {
        "result_key": "gmm_full_best",
        "tuning_key": "tuning_gmm_full",
        "tuning_secondary": ("BIC", "bics"),
        "tuning_label": "BIC",
    },
    "GMM (diag covariance)": {
        "result_key": "gmm_diag_best",
        "tuning_key": "tuning_gmm_diag",
        "tuning_secondary": ("BIC", "bics"),
        "tuning_label": "BIC",
    },
}


st.title("Cluster Explorer")
st.caption("K-Means and GMM clustering on the 12D standardized feature space (precomputed)")

with st.sidebar:
    st.header("Clustering Controls")
    algorithm = st.selectbox("Algorithm", list(ALGO_OPTIONS.keys()))

algo_cfg = ALGO_OPTIONS[algorithm]
cluster_result: ClusterResult = artifacts[algo_cfg["result_key"]]
tuning: TuningResult = artifacts[algo_cfg["tuning_key"]]
k_value = cluster_result.n_clusters

with st.sidebar:
    st.info(f"Best K = {k_value} (selected by {algo_cfg['tuning_label']})")

tab_viz, tab_tuning, tab_profile, tab_metrics = st.tabs(
    ["Cluster Visualization", "Hyperparameter Tuning", "Cluster Profiling", "Evaluation Metrics"]
)

# ── Tab 1: Visualization ─────────────────────────────────────────────────────

with tab_viz:
    fig = _scatter_plot(pca_2d, cluster_result.labels, df_encoded, f"{algorithm} Clusters (PCA)", "PCA")
    st.plotly_chart(fig, width="stretch", key="cluster_scatter")

    dist = pd.Series(cluster_result.labels).value_counts().sort_index().reset_index()
    dist.columns = ["cluster", "count"]
    dist["cluster"] = dist["cluster"].astype(str)
    dist_fig = px.bar(dist, x="cluster", y="count", color="cluster", title="Cluster Size Distribution")
    dist_fig.update_layout(showlegend=False, margin={"l": 20, "r": 20, "t": 50, "b": 20})
    st.plotly_chart(dist_fig, width="stretch", key="cluster_dist")

# ── Tab 2: Tuning ────────────────────────────────────────────────────────────

with tab_tuning:
    st.subheader(f"{algorithm} Hyperparameter Tuning")

    col1, col2 = st.columns(2)
    with col1:
        if tuning.silhouette_scores:
            sil_fig = _tuning_elbow_chart(tuning, "Silhouette Score", tuning.silhouette_scores)
            st.plotly_chart(sil_fig, width="stretch", key="tuning_sil")
        else:
            st.caption("Silhouette not computed for GMM tuning (BIC is the principled criterion).")

    with col2:
        sec_label, sec_attr = algo_cfg["tuning_secondary"]
        sec_values = getattr(tuning, sec_attr)
        if sec_values:
            sec_fig = _tuning_elbow_chart(tuning, sec_label, sec_values)
            st.plotly_chart(sec_fig, width="stretch", key="tuning_secondary")

    st.success(f"Recommended K = {tuning.best_k}")

# ── Tab 3: Profiling ─────────────────────────────────────────────────────────

with tab_profile:
    st.subheader("Cluster Profiles")

    means_df = _cluster_feature_means(df_encoded, cluster_result.labels)
    top_genres = _cluster_top_genres(df_encoded, cluster_result.labels)

    auto_labels = {cid: _auto_label_cluster(means_df.loc[cid]) for cid in means_df.index}

    summary_rows = []
    for cid in sorted(means_df.index):
        count = int((cluster_result.labels == cid).sum())
        genres_str = ", ".join(f"{g} ({c})" for g, c in top_genres.get(cid, [])[:3])
        summary_rows.append({
            "Cluster": cid,
            "Size": count,
            "Label": auto_labels[cid],
            "Top Genres": genres_str,
        })
    st.dataframe(pd.DataFrame(summary_rows), width="stretch")

    display_cols = {c: DISPLAY_NAMES.get(c, c) for c in means_df.columns}
    means_display = means_df.rename(columns=display_cols)

    heatmap_fig = px.imshow(
        means_display.T,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        title="Mean Feature Values per Cluster",
        labels={"x": "Cluster", "y": "Feature"},
    )
    heatmap_fig.update_layout(margin={"l": 20, "r": 20, "t": 50, "b": 20})
    st.plotly_chart(heatmap_fig, width="stretch", key="feature_heatmap")

    selected_cluster = st.selectbox(
        "Inspect cluster",
        options=sorted(int(c) for c in np.unique(cluster_result.labels)),
        key="inspect_cluster",
    )
    st.markdown(f"**Auto-label**: {auto_labels[selected_cluster]}")
    st.markdown("**Top genres**:")
    for genre, count in top_genres.get(selected_cluster, []):
        st.markdown(f"- {genre} ({count} songs)")

    cluster_songs = df_encoded[cluster_result.labels == selected_cluster].copy()
    cluster_songs = cluster_songs.sort_values("popularity", ascending=False)
    visible = [c for c in ["track_name", "artists", "track_genre", "popularity", "energy", "valence", "danceability"] if c in cluster_songs.columns]
    st.dataframe(cluster_songs[visible].head(30).reset_index(drop=True), width="stretch")

# ── Tab 4: Metrics ────────────────────────────────────────────────────────────

with tab_metrics:
    st.subheader("Cluster Evaluation Metrics")

    all_metrics: list[ClusterMetrics] = artifacts["metrics_comparison"]
    comparison_df = metrics_comparison_table(all_metrics)
    st.dataframe(comparison_df, width="stretch")

    st.caption(
        "**Note**: Genre labels are imperfect ground truth. Low ARI/NMI does not necessarily mean bad clusters "
        "— it may mean clustering captured acoustic structure that genre labels don't reflect."
    )

    with st.expander("Metric definitions"):
        st.markdown("""
- **Silhouette Score** [-1, 1]: Measures intra-cluster cohesion vs. inter-cluster separation. Higher is better.
- **Davies-Bouldin Index** [0, +inf]: Ratio of within-cluster scatter to between-cluster separation. Lower is better.
- **Calinski-Harabasz Index** [0, +inf]: Ratio of between-cluster variance to within-cluster variance. Higher is better.
- **ARI (Adjusted Rand Index)** [-1, 1]: Agreement between clusters and genre labels, adjusted for chance. Higher is better.
- **NMI (Normalized Mutual Information)** [0, 1]: Shared information between clusters and genre labels. Higher is better.
        """)
