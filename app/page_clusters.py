"""Phase 2: Cluster visualization, profiling, and evaluation page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from umap import UMAP

from src.clustering import (
    ClusterResult,
    TuningResult,
    fit_gmm,
    fit_kmeans,
    tune_gmm,
    tune_kmeans,
)
from src.evaluate import ClusterMetrics, evaluate_clustering, metrics_comparison_table
from src.explain import DISPLAY_NAMES
from src.features import FEATURE_COLUMNS_ENCODED, build_feature_matrix, load_dataset


# ── Data loading (cached) ───────────────────────────────────────────────────


@st.cache_resource
def _load_data():
    df = load_dataset()
    feature_matrix, scaler, df_encoded = build_feature_matrix(df)
    return df_encoded, feature_matrix


@st.cache_resource
def _compute_projections(_feature_matrix: np.ndarray):
    pca_2d = PCA(n_components=2, random_state=42).fit_transform(_feature_matrix)
    umap_2d = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3).fit_transform(_feature_matrix)
    return pca_2d, umap_2d


@st.cache_resource
def _tune_kmeans(_feature_matrix: np.ndarray):
    return tune_kmeans(_feature_matrix, k_range=range(5, 31))


@st.cache_resource
def _tune_gmm(_feature_matrix: np.ndarray):
    return tune_gmm(_feature_matrix, k_range=range(5, 31))


# ── Cluster profiling helpers ────────────────────────────────────────────────


def _cluster_feature_means(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean feature values per cluster."""
    df_work = df.copy()
    df_work["cluster"] = labels
    feature_cols = [c for c in FEATURE_COLUMNS_ENCODED if c in df_work.columns]
    return df_work.groupby("cluster")[feature_cols].mean()


def _cluster_top_genres(df: pd.DataFrame, labels: np.ndarray, top_n: int = 5) -> dict[int, list[tuple[str, int]]]:
    """Top genres per cluster by frequency."""
    df_work = df.copy()
    df_work["cluster"] = labels
    result = {}
    for cid in sorted(df_work["cluster"].unique()):
        genre_counts = df_work[df_work["cluster"] == cid]["track_genre"].value_counts().head(top_n)
        result[int(cid)] = list(zip(genre_counts.index.tolist(), genre_counts.values.tolist()))
    return result


def _auto_label_cluster(mean_row: pd.Series) -> str:
    """Generate a short human-readable label from dominant features."""
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


def _tuning_elbow_chart(tuning: TuningResult, metric_name: str, values: list[float], lower_is_better: bool = False) -> go.Figure:
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


df_encoded, feature_matrix = _load_data()

st.title("Cluster Explorer")
st.caption("K-Means and GMM clustering on the 12D standardized feature space")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Clustering Controls")
    algorithm = st.selectbox("Algorithm", ["K-Means", "GMM"])
    auto_k = st.checkbox("Auto-select best K", value=True)

    if auto_k:
        with st.spinner("Tuning hyperparameters (this may take a moment)..."):
            if algorithm == "K-Means":
                tuning = _tune_kmeans(feature_matrix)
            else:
                tuning = _tune_gmm(feature_matrix)
        k_value = tuning.best_k
        st.info(f"Best K = {k_value} (by {'silhouette' if algorithm == 'K-Means' else 'BIC'})")
    else:
        tuning = None
        k_value = st.slider("Number of clusters (K)", min_value=5, max_value=30, value=10)

# --- Fit the selected model ---
with st.spinner(f"Fitting {algorithm} with K={k_value}..."):
    if algorithm == "K-Means":
        cluster_result = fit_kmeans(feature_matrix, n_clusters=k_value)
    else:
        cluster_result = fit_gmm(feature_matrix, n_clusters=k_value)

    metrics = evaluate_clustering(
        algorithm=cluster_result.algorithm,
        n_clusters=cluster_result.n_clusters,
        feature_matrix=feature_matrix,
        labels=cluster_result.labels,
        genre_labels=df_encoded["track_genre"],
    )

# --- Tabs ---
tab_viz, tab_tuning, tab_profile, tab_metrics = st.tabs(
    ["Cluster Visualization", "Hyperparameter Tuning", "Cluster Profiling", "Evaluation Metrics"]
)

# ── Tab 1: Visualization ─────────────────────────────────────────────────────

with tab_viz:
    with st.spinner("Computing projections..."):
        pca_2d, umap_2d = _compute_projections(feature_matrix)

    proj_method = st.radio("Projection method", ["PCA", "UMAP"], horizontal=True, key="proj_method")

    if proj_method == "PCA":
        fig = _scatter_plot(pca_2d, cluster_result.labels, df_encoded, f"{algorithm} Clusters (PCA)", "PCA")
    else:
        fig = _scatter_plot(umap_2d, cluster_result.labels, df_encoded, f"{algorithm} Clusters (UMAP)", "UMAP")

    st.plotly_chart(fig, width="stretch", key="cluster_scatter")

    # Distribution bar chart
    dist = pd.Series(cluster_result.labels).value_counts().sort_index().reset_index()
    dist.columns = ["cluster", "count"]
    dist["cluster"] = dist["cluster"].astype(str)
    dist_fig = px.bar(dist, x="cluster", y="count", color="cluster", title="Cluster Size Distribution")
    dist_fig.update_layout(showlegend=False, margin={"l": 20, "r": 20, "t": 50, "b": 20})
    st.plotly_chart(dist_fig, width="stretch", key="cluster_dist")

# ── Tab 2: Tuning ────────────────────────────────────────────────────────────

with tab_tuning:
    st.subheader(f"{algorithm} Hyperparameter Tuning")

    if tuning is None:
        st.info("Enable 'Auto-select best K' in the sidebar to see tuning charts.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            sil_fig = _tuning_elbow_chart(tuning, "Silhouette Score", tuning.silhouette_scores)
            st.plotly_chart(sil_fig, width="stretch", key="tuning_sil")

        with col2:
            if algorithm == "K-Means" and tuning.inertias:
                inertia_fig = _tuning_elbow_chart(tuning, "Inertia (Elbow)", tuning.inertias, lower_is_better=True)
                st.plotly_chart(inertia_fig, width="stretch", key="tuning_inertia")
            elif algorithm == "GMM" and tuning.bics:
                bic_fig = _tuning_elbow_chart(tuning, "BIC", tuning.bics, lower_is_better=True)
                st.plotly_chart(bic_fig, width="stretch", key="tuning_bic")

        st.success(f"Recommended K = {tuning.best_k}")

# ── Tab 3: Profiling ─────────────────────────────────────────────────────────

with tab_profile:
    st.subheader("Cluster Profiles")

    means_df = _cluster_feature_means(df_encoded, cluster_result.labels)
    top_genres = _cluster_top_genres(df_encoded, cluster_result.labels)

    # Auto labels
    auto_labels = {cid: _auto_label_cluster(means_df.loc[cid]) for cid in means_df.index}

    # Summary table
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

    # Feature heatmap
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

    # Inspect individual cluster
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

    # Run both algorithms for comparison
    @st.cache_resource
    def _evaluate_both(_feature_matrix, _genre_labels, k):
        km_result = fit_kmeans(_feature_matrix, n_clusters=k)
        km_metrics = evaluate_clustering("K-Means", k, _feature_matrix, km_result.labels, _genre_labels)
        gmm_result = fit_gmm(_feature_matrix, n_clusters=k)
        gmm_metrics = evaluate_clustering("GMM", k, _feature_matrix, gmm_result.labels, _genre_labels)
        return [km_metrics, gmm_metrics]

    all_metrics = _evaluate_both(feature_matrix, df_encoded["track_genre"], k_value)
    comparison_df = metrics_comparison_table(all_metrics)
    st.dataframe(comparison_df, width="stretch")

    st.caption(
        "**Note**: Genre labels are imperfect ground truth. Low ARI/NMI does not necessarily mean bad clusters "
        "— it may mean clustering captured acoustic structure that genre labels don't reflect."
    )

    # Metric explanations
    with st.expander("Metric definitions"):
        st.markdown("""
- **Silhouette Score** [-1, 1]: Measures intra-cluster cohesion vs. inter-cluster separation. Higher is better.
- **Davies-Bouldin Index** [0, +inf]: Ratio of within-cluster scatter to between-cluster separation. Lower is better.
- **Calinski-Harabasz Index** [0, +inf]: Ratio of between-cluster variance to within-cluster variance. Higher is better.
- **ARI (Adjusted Rand Index)** [-1, 1]: Agreement between clusters and genre labels, adjusted for chance. Higher is better.
- **NMI (Normalized Mutual Information)** [0, 1]: Shared information between clusters and genre labels. Higher is better.
        """)
