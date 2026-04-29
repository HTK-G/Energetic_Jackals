"""Model Analysis page.

Combines clustering analysis (hyperparameter tuning, cluster profiling,
internal metrics, cluster overlap) with recommendation-engine evaluation
(6 methods × 2 metrics).  All training is precomputed offline by
`scripts/precompute.py` and `scripts/evaluate_recommendations.py`; this
page only loads pickled artifacts.
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
    "tuning_kmeans.joblib",
    "tuning_gmm_full.joblib",
    "tuning_gmm_diag.joblib",
    "kmeans_best.joblib",
    "gmm_full_best.joblib",
    "gmm_diag_best.joblib",
    "metrics_comparison.joblib",
]
RECOMMENDATION_EVAL_ARTIFACT = "recommendation_eval.joblib"


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
    out = {name.removesuffix(".joblib"): joblib.load(ARTIFACTS_DIR / name) for name in REQUIRED_ARTIFACTS}
    rec_eval_path = ARTIFACTS_DIR / RECOMMENDATION_EVAL_ARTIFACT
    out["recommendation_eval"] = joblib.load(rec_eval_path) if rec_eval_path.exists() else None
    return out


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


st.title("Model Analysis")
st.caption(
    "Hyperparameter tuning, cluster profiling, internal metrics, and "
    "recommendation-engine evaluation — all on precomputed artifacts."
)

with st.sidebar:
    st.header("Clustering Controls")
    algorithm = st.selectbox("Algorithm", list(ALGO_OPTIONS.keys()))

algo_cfg = ALGO_OPTIONS[algorithm]
cluster_result: ClusterResult = artifacts[algo_cfg["result_key"]]
tuning: TuningResult = artifacts[algo_cfg["tuning_key"]]
k_value = cluster_result.n_clusters

with st.sidebar:
    st.info(f"Best K = {k_value} (selected by {algo_cfg['tuning_label']})")

tab_tuning, tab_profile, tab_metrics, tab_eval = st.tabs(
    ["Hyperparameter Tuning", "Cluster Profiling", "Internal Metrics", "Recommendation Evaluation"]
)

# ── Tab 1: Tuning ────────────────────────────────────────────────────────────

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

# ── Tab 2: Profiling ─────────────────────────────────────────────────────────

with tab_profile:
    st.subheader("Cluster Profiles")

    means_df = _cluster_feature_means(df_encoded, cluster_result.labels)
    top_genres = _cluster_top_genres(df_encoded, cluster_result.labels)

    auto_labels = {cid: _auto_label_cluster(means_df.loc[cid]) for cid in means_df.index}

    # Cluster size distribution (was in the dropped Visualization tab)
    dist = pd.Series(cluster_result.labels).value_counts().sort_index().reset_index()
    dist.columns = ["cluster", "count"]
    dist["cluster"] = dist["cluster"].astype(str)
    dist_fig = px.bar(dist, x="cluster", y="count", color="cluster",
                      title="Cluster Size Distribution")
    dist_fig.update_layout(showlegend=False, margin={"l": 20, "r": 20, "t": 50, "b": 20})
    st.plotly_chart(dist_fig, width="stretch", key="cluster_dist")

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

# ── Tab 3: Internal Metrics + Cluster Overlap ────────────────────────────────

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

    # Cluster overlap heatmap — K-Means × GMM-full contingency
    st.subheader("Cluster Overlap (K-Means × GMM-full)")
    st.caption(
        "If both algorithms find the same partition under different labels, the matrix "
        "(after row-permutation) would be near-block-diagonal.  A diffuse matrix means the "
        "two algorithms found different structure."
    )
    km_labels = artifacts["kmeans_best"].labels
    gmm_labels = artifacts["gmm_full_best"].labels
    cm = pd.crosstab(
        pd.Series(km_labels, name="K-Means cluster"),
        pd.Series(gmm_labels, name="GMM-full cluster"),
    )
    overlap_fig = px.imshow(
        cm.values,
        x=cm.columns.tolist(), y=cm.index.tolist(),
        labels={"x": "GMM-full cluster", "y": "K-Means cluster", "color": "song count"},
        color_continuous_scale="Oranges",
        aspect="auto",
        title=f"Contingency matrix — {cm.shape[0]} K-Means × {cm.shape[1]} GMM-full",
    )
    overlap_fig.update_layout(margin={"l": 20, "r": 20, "t": 50, "b": 20}, height=480)
    st.plotly_chart(overlap_fig, width="stretch", key="cluster_overlap")


# ── Tab 4: Recommendation Evaluation ─────────────────────────────────────────

with tab_eval:
    st.subheader("Recommendation Engine — 6-method comparison")
    st.caption(
        "Three of our methods (KNN, K-Means cluster, GMM posterior) versus three "
        "trivial baselines (random, popularity, genre-match) on 500 query songs.  "
        "Source: `scripts/evaluate_recommendations.py`."
    )

    eval_payload = artifacts.get("recommendation_eval")
    if eval_payload is None:
        st.warning(
            "`artifacts/recommendation_eval.joblib` not found.  Run "
            "`uv run python -m scripts.evaluate_recommendations` to generate it."
        )
    else:
        with st.expander("Methodology", expanded=False):
            st.markdown("""
**Methods.** Three baselines + three of ours.

| Method | Definition |
|---|---|
| `random` | Uniform random K from the catalog |
| `popularity` | Top-K most popular songs (same for every query — cold-start fallback) |
| `genre_match` | Random K from the same `track_genre` as the query |
| `knn` | `RecommendationEngine.recommend` — full-feature cosine KNN |
| `kmeans_cluster` | `RecommendationEngine.recommend_by_cluster` — best K-Means model |
| `gmm_posterior` | `RecommendationEngine.recommend_by_gmm` — best GMM-full model |

**Metrics.**

- **Genre Hit Rate** [0, 1] — fraction of top-K recs sharing at least one
  label with the query's `all_genres` set.  Higher = "looks like the query."
- **Genre Coverage** [1, K] — number of distinct `track_genre` values in
  the top-K.  Higher = "spans genres" (good when genre labels are heterogeneous).
            """)

        results = eval_payload["results"]
        n_queries = eval_payload["n_queries"]
        top_k = eval_payload["top_k"]

        st.markdown(f"**{n_queries} queries × {top_k} recommendations** (random_state="
                    f"{eval_payload['random_state']})")

        # Comparison table
        order = ["random", "popularity", "genre_match", "knn", "kmeans_cluster", "gmm_posterior"]
        rows = []
        for name in order:
            r = results[name]
            rows.append({
                "Method": name,
                "Hit Rate (μ)": round(r["hit_rate_mean"], 4),
                "Hit Rate (σ)": round(r["hit_rate_std"], 4),
                "Coverage (μ)": round(r["coverage_mean"], 2),
                "Coverage (σ)": round(r["coverage_std"], 2),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")

        # Box plots — hit rate and coverage
        raw = eval_payload["raw_scores"]
        col_hr, col_cov = st.columns(2)

        hr_long = pd.DataFrame({
            "Method": np.repeat(order, [len(raw[m]["hit_rates"]) for m in order]),
            "Hit Rate": np.concatenate([raw[m]["hit_rates"] for m in order]),
        })
        hr_fig = px.box(hr_long, x="Method", y="Hit Rate",
                        title="Genre Hit Rate", color="Method",
                        color_discrete_sequence=px.colors.sequential.Greens_r)
        hr_fig.update_layout(showlegend=False, margin={"l": 20, "r": 20, "t": 50, "b": 20})
        col_hr.plotly_chart(hr_fig, width="stretch", key="eval_hr_box")

        cov_long = pd.DataFrame({
            "Method": np.repeat(order, [len(raw[m]["coverages"]) for m in order]),
            "Coverage": np.concatenate([raw[m]["coverages"] for m in order]),
        })
        cov_fig = px.box(cov_long, x="Method", y="Coverage",
                         title="Genre Coverage", color="Method",
                         color_discrete_sequence=px.colors.sequential.Blues_r)
        cov_fig.update_layout(showlegend=False, margin={"l": 20, "r": 20, "t": 50, "b": 20})
        col_cov.plotly_chart(cov_fig, width="stretch", key="eval_cov_box")

        # Trade-off scatter: hit_rate vs coverage with mean ± std error bars
        scatter_df = pd.DataFrame([
            {
                "Method": name,
                "Hit Rate": results[name]["hit_rate_mean"],
                "Coverage": results[name]["coverage_mean"],
                "HR_std": results[name]["hit_rate_std"],
                "Cov_std": results[name]["coverage_std"],
            }
            for name in order
        ])
        scatter_fig = px.scatter(
            scatter_df, x="Hit Rate", y="Coverage", text="Method",
            error_x="HR_std", error_y="Cov_std",
            title="Hit Rate vs Coverage trade-off (mean ± σ)",
        )
        scatter_fig.update_traces(textposition="top center", marker=dict(size=12, color="#c8956c"))
        scatter_fig.update_layout(margin={"l": 20, "r": 20, "t": 50, "b": 20}, height=480)
        st.plotly_chart(scatter_fig, width="stretch", key="eval_tradeoff")

        st.markdown(f"""
**Reading the trade-off.**

- `random` ({results['random']['hit_rate_mean']:.3f} hit rate /
  {results['random']['coverage_mean']:.1f} coverage) and
  `popularity` ({results['popularity']['hit_rate_mean']:.3f} /
  {results['popularity']['coverage_mean']:.1f}) sit at the bottom-left:
  trivial baselines neither match the query genre nor span the catalog
  (popularity always returns the same songs).
- `genre_match` (1.000 / 1.0) sits at the top-left of the trade-off:
  perfect hit rate by construction, but every rec has the same genre as
  the query.
- Our three methods sit in the middle-right: hit rates of
  {results['knn']['hit_rate_mean']:.3f}–{results['kmeans_cluster']['hit_rate_mean']:.3f}
  (≈ {results['knn']['hit_rate_mean']/results['random']['hit_rate_mean']:.0f}× random)
  with coverage {min(results[m]['coverage_mean'] for m in ['knn', 'kmeans_cluster', 'gmm_posterior']):.1f}–{max(results[m]['coverage_mean'] for m in ['knn', 'kmeans_cluster', 'gmm_posterior']):.1f}.
  This is the report's central finding: feature-based recommendation
  spans genres while staying relevant.
- `gmm_posterior` is the highest-coverage of our three at the cost of
  some hit rate — soft cluster membership pulls in songs from more
  genres than hard cluster assignment does.
        """)
