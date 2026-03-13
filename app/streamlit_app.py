"""Streamlit frontend for the Spotify ML project."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data.loader import detect_dataset_path, load_spotify_dataset, standardize_spotify_schema
from src.data.preprocessing import PreparedData, preprocess_spotify_dataframe
from src.evaluation.metrics import summarize_pipeline_metrics
from src.models.embeddings import EmbeddingResult, compute_song_embeddings
from src.models.kmeans import NumpyKMeans
from src.recommender.recommend import RecommendationEngine
from src.visualization.plots import (
    build_cluster_distribution_plot,
    build_correlation_heatmap,
    build_embedding_scatter,
    build_radar_chart,
)

DEFAULT_DATASET_PATH = "data/raw/spotify_audio_features.csv"


def _resolve_source_label(dataset_path: str | None, uploaded_bytes: bytes | None) -> str:
    if uploaded_bytes:
        return "Uploaded CSV"
    if dataset_path and Path(dataset_path).expanduser().exists():
        return str(Path(dataset_path).expanduser())
    detected_path = detect_dataset_path()
    if detected_path is not None:
        return str(detected_path)
    return "Generated demo dataset"


def _load_catalog(dataset_path: str | None, uploaded_bytes: bytes | None, sample_size: int) -> pd.DataFrame:
    if uploaded_bytes:
        frame = pd.read_csv(io.BytesIO(uploaded_bytes))
        frame = standardize_spotify_schema(frame)
        if len(frame) > sample_size:
            frame = frame.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            frame = frame.reset_index(drop=True)
        return frame

    resolved_path = dataset_path or None
    return load_spotify_dataset(dataset_path=resolved_path, sample_size=sample_size)


def build_pipeline(
    dataset_path: str | None,
    uploaded_bytes: bytes | None,
    sample_size: int,
    embedding_method: str,
    embedding_dim: int,
    cluster_count: int,
    autoencoder_epochs: int,
) -> dict[str, Any]:
    """Load data, preprocess features, compute embeddings, and fit clustering."""
    catalog = _load_catalog(dataset_path, uploaded_bytes, sample_size)
    prepared_data: PreparedData = preprocess_spotify_dataframe(catalog)

    embedding_result: EmbeddingResult = compute_song_embeddings(
        prepared_data.feature_matrix,
        method=embedding_method,
        embedding_dim=embedding_dim,
        autoencoder_epochs=autoencoder_epochs,
    )

    effective_cluster_count = max(2, min(cluster_count, len(prepared_data.cleaned_frame)))
    kmeans_model = NumpyKMeans(n_clusters=effective_cluster_count, random_state=42)
    cluster_labels = kmeans_model.fit_predict(embedding_result.embeddings)

    enriched_catalog = prepared_data.cleaned_frame.copy()
    enriched_catalog["cluster"] = cluster_labels
    enriched_catalog["embedding_x"] = embedding_result.visualization_projection[:, 0]
    enriched_catalog["embedding_y"] = embedding_result.visualization_projection[:, 1]

    metrics = summarize_pipeline_metrics(
        feature_matrix=prepared_data.feature_matrix,
        embeddings=embedding_result.embeddings,
        labels=cluster_labels,
    )
    engine = RecommendationEngine(
        catalog=enriched_catalog,
        embeddings=embedding_result.embeddings,
        cluster_labels=cluster_labels,
    )

    return {
        "catalog": enriched_catalog,
        "prepared_data": prepared_data,
        "embedding_result": embedding_result,
        "cluster_labels": cluster_labels,
        "kmeans_model": kmeans_model,
        "metrics": metrics,
        "engine": engine,
        "source_label": _resolve_source_label(dataset_path, uploaded_bytes),
    }


def _song_option_label(engine: RecommendationEngine, index: int) -> str:
    return engine.song_label(index)


def _build_sidebar_controls() -> tuple[bool, dict[str, Any]]:
    with st.sidebar:
        st.header("Pipeline Controls")
        with st.form("pipeline_controls"):
            dataset_path = st.text_input("Dataset path", value=DEFAULT_DATASET_PATH)
            uploaded_file = st.file_uploader("Upload Spotify CSV", type=["csv"])
            sample_size = st.number_input(
                "Sample size",
                min_value=200,
                max_value=130000,
                value=5000,
                step=500,
            )
            embedding_method = st.selectbox(
                "Embedding method",
                options=["pca", "autoencoder"],
                format_func=lambda value: value.upper(),
            )
            embedding_dim = st.slider("Embedding dimension", min_value=2, max_value=16, value=12)
            cluster_count = st.slider("Cluster count", min_value=2, max_value=20, value=8)
            autoencoder_epochs = st.slider("Autoencoder epochs", min_value=5, max_value=40, value=15)
            submit = st.form_submit_button("Run pipeline")

        st.caption(
            "Source dataset: Kaggle Spotify Audio Features. If no CSV is available, the app uses a generated demo catalog."
        )

    return submit, {
        "dataset_path": dataset_path,
        "uploaded_bytes": uploaded_file.getvalue() if uploaded_file is not None else None,
        "sample_size": int(sample_size),
        "embedding_method": embedding_method,
        "embedding_dim": int(embedding_dim),
        "cluster_count": int(cluster_count),
        "autoencoder_epochs": int(autoencoder_epochs),
    }


def _initialize_pipeline(default_controls: dict[str, Any]) -> None:
    if "pipeline_state" in st.session_state:
        return

    with st.spinner("Building initial Spotify pipeline..."):
        st.session_state["pipeline_state"] = build_pipeline(**default_controls)


def main() -> None:
    st.set_page_config(page_title="Spotify ML Project", layout="wide")
    st.title("Spotify Audio Features Lab")
    st.caption(
        "Explore the Kaggle Spotify Audio Features dataset with representation learning, scratch K-Means clustering, and similarity search."
    )

    submitted, controls = _build_sidebar_controls()
    _initialize_pipeline(controls)

    if submitted:
        with st.spinner("Rebuilding embeddings, clusters, and recommendations..."):
            st.session_state["pipeline_state"] = build_pipeline(**controls)

    pipeline_state = st.session_state["pipeline_state"]
    catalog: pd.DataFrame = pipeline_state["catalog"]
    prepared_data: PreparedData = pipeline_state["prepared_data"]
    embedding_result: EmbeddingResult = pipeline_state["embedding_result"]
    cluster_labels = pipeline_state["cluster_labels"]
    metrics = pipeline_state["metrics"]
    engine: RecommendationEngine = pipeline_state["engine"]

    summary_columns = st.columns(5)
    summary_columns[0].metric("Songs", f"{len(catalog):,}")
    summary_columns[1].metric("Embedding Method", embedding_result.method.upper())
    summary_columns[2].metric("Embedding Dim", embedding_result.embeddings.shape[1])
    summary_columns[3].metric("Clusters", int(metrics["cluster_count"]))
    summary_columns[4].metric("Data Source", pipeline_state["source_label"])

    tab_explorer, tab_recommendation, tab_clusters, tab_embeddings = st.tabs(
        ["Song Explorer", "Recommendation", "Clustering Visualization", "Embedding Explorer"]
    )

    with tab_explorer:
        search_query = st.text_input("Search songs or artists", placeholder="Type a song or artist name")
        matches = engine.search_songs(search_query, limit=25)

        if matches.empty:
            st.info("No matching songs were found for the current query.")
        else:
            selected_index = st.selectbox(
                "Select a song",
                options=matches.index.tolist(),
                format_func=lambda index: _song_option_label(engine, index),
                key="explorer_song_selection",
            )
            selected_song = catalog.loc[selected_index]

            meta_column, chart_column = st.columns([1.0, 1.3])
            with meta_column:
                metadata_columns = [
                    column
                    for column in [
                        "song_name",
                        "artist_name",
                        "album_name",
                        "playlist_genre",
                        "cluster",
                        "popularity",
                        "duration_ms",
                        "tempo",
                    ]
                    if column in selected_song.index
                ]
                metadata_frame = selected_song[metadata_columns].to_frame(name="value")
                st.dataframe(metadata_frame, use_container_width=True)

            with chart_column:
                radar_figure = build_radar_chart(selected_song, catalog)
                st.plotly_chart(radar_figure, use_container_width=True)

    with tab_recommendation:
        selected_index = st.selectbox(
            "Seed song",
            options=catalog.index.tolist(),
            format_func=lambda index: _song_option_label(engine, index),
            key="recommendation_song_selection",
        )
        recommendation_mode = st.radio(
            "Recommendation mode",
            options=["embedding", "cluster-based"],
            horizontal=True,
        )
        top_k = st.slider("Top-k similar songs", min_value=3, max_value=20, value=10)

        if recommendation_mode == "embedding":
            recommendation_frame = engine.recommend_by_embedding(selected_index, top_k=top_k)
        else:
            recommendation_frame = engine.recommend_by_cluster(selected_index, top_k=top_k)

        st.dataframe(recommendation_frame, use_container_width=True)

    with tab_clusters:
        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Avg Intra-Cluster Distance",
            f"{metrics['average_intra_cluster_distance']:.3f}",
        )
        silhouette_value = metrics["silhouette_score"]
        metric_columns[1].metric("Silhouette", "N/A" if silhouette_value is None else f"{silhouette_value:.3f}")
        distance_correlation = metrics["distance_correlation"]
        metric_columns[2].metric(
            "Distance Correlation",
            "N/A" if distance_correlation is None else f"{distance_correlation:.3f}",
        )
        neighbor_overlap = metrics["neighbor_overlap"]
        metric_columns[3].metric(
            "Neighbor Overlap",
            "N/A" if neighbor_overlap is None else f"{neighbor_overlap:.3f}",
        )

        cluster_chart_column, cluster_distribution_column = st.columns(2)
        with cluster_chart_column:
            cluster_figure = build_embedding_scatter(
                catalog,
                embedding_result.visualization_projection,
                cluster_labels=cluster_labels,
                title="Cluster Map",
            )
            st.plotly_chart(cluster_figure, use_container_width=True)

        with cluster_distribution_column:
            distribution_figure = build_cluster_distribution_plot(cluster_labels)
            st.plotly_chart(distribution_figure, use_container_width=True)

        selected_cluster = st.selectbox(
            "Inspect cluster",
            options=sorted(int(cluster_id) for cluster_id in pd.Series(cluster_labels).unique()),
        )
        cluster_members = catalog.loc[catalog["cluster"] == selected_cluster].copy()
        cluster_members = cluster_members.sort_values(by="popularity", ascending=False)
        visible_columns = [
            column
            for column in ["song_name", "artist_name", "playlist_genre", "popularity", "tempo", "energy", "valence"]
            if column in cluster_members.columns
        ]
        st.dataframe(cluster_members[visible_columns].reset_index(drop=True), use_container_width=True)

    with tab_embeddings:
        embedding_figure = build_embedding_scatter(
            catalog,
            embedding_result.visualization_projection,
            cluster_labels=None,
            title="Embedding Explorer",
        )
        st.plotly_chart(embedding_figure, use_container_width=True)

        correlation_figure = build_correlation_heatmap(
            prepared_data.cleaned_frame,
            feature_columns=prepared_data.continuous_features,
        )
        st.plotly_chart(correlation_figure, use_container_width=True)

        sample_preview = catalog[[column for column in ["song_name", "artist_name", "cluster", "embedding_x", "embedding_y"] if column in catalog.columns]].head(20)
        st.dataframe(sample_preview.reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()
