"""Phase 1 & 2: Song search and recommendation page with cluster-aware modes."""

from __future__ import annotations

import streamlit as st

from src.clustering import fit_gmm, fit_kmeans
from src.features import FEATURE_COLUMNS_ENCODED, build_feature_matrix, load_dataset
from src.recommend import RecommendationEngine
from src.explain import build_comparison_radar, build_single_radar, explain_recommendation


@st.cache_resource
def _build_engine():
    """Load data, build features, and create the recommendation engine."""
    df = load_dataset()
    feature_matrix, scaler, df_encoded = build_feature_matrix(df)
    engine = RecommendationEngine(df_encoded, feature_matrix)
    return engine, df_encoded, feature_matrix


@st.cache_resource
def _fit_clusters(_feature_matrix, k: int):
    """Cache clustering results."""
    km = fit_kmeans(_feature_matrix, n_clusters=k)
    gmm = fit_gmm(_feature_matrix, n_clusters=k)
    return km, gmm


engine, df_encoded, feature_matrix = _build_engine()

st.title("Song Search & Recommend")
st.caption(f"{len(df_encoded):,} songs loaded from Spotify dataset")

# --- Song Search ---
search_query = st.text_input("Search for a song or artist", placeholder="e.g. Blinding Lights, Drake, ...")

search_results = None
if search_query.strip():
    matches = engine.search_songs(search_query, limit=20)
    if matches.empty:
        st.warning("No matching songs found. Try a different query.")
    else:
        st.dataframe(matches.reset_index(drop=True), use_container_width=True)
        search_results = matches

# --- Song Selection ---
st.divider()
st.subheader("Get Recommendations")

# Two ways to select: from search results, or from full catalog
selection_mode = st.radio(
    "Select seed song from:",
    options=["Search results", "Full catalog"],
    horizontal=True,
    index=0 if search_results is not None and len(search_results) > 0 else 1,
)

if selection_mode == "Search results" and search_results is not None and len(search_results) > 0:
    selected_index = st.selectbox(
        "Pick from search results",
        options=search_results.index.tolist(),
        format_func=lambda i: engine.song_label(i),
        key="search_select",
    )
else:
    song_labels = [engine.song_label(i) for i in range(len(engine.df))]
    selected_index = st.selectbox(
        "Select a seed song",
        options=range(len(engine.df)),
        format_func=lambda i: song_labels[i],
        key="catalog_select",
    )

# Recommendation mode
rec_mode = st.radio(
    "Recommendation mode",
    options=["Embedding (KNN)", "K-Means cluster", "GMM posterior"],
    horizontal=True,
)

# --- NEW: Reranking UI (Demo only) ---
st.subheader("Reranking Strategy (Experimental)")

rerank_mode = st.radio(
    "Reranking",
    options=["Default", "Feature-aware (Auto)", "Feature-aware (Manual)"],
    horizontal=True,
)

# Manual controls (only show if selected)
if rerank_mode == "Feature-aware (Manual)":
    st.caption("Adjust feature preferences:")

    weight_energy = st.slider("Energy importance", 0.0, 2.0, 1.0)
    weight_tempo = st.slider("Tempo importance", 0.0, 2.0, 1.0)
    weight_acoustic = st.slider("Acousticness importance", 0.0, 2.0, 1.0)

top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=10)

if rec_mode in ("K-Means cluster", "GMM posterior"):
    cluster_k = st.slider("Number of clusters (K)", min_value=5, max_value=30, value=10, key="rec_cluster_k")

if st.button("Recommend", type="primary"):
    query_row = engine.df.iloc[selected_index]
    st.markdown(f"**Query song**: {query_row['track_name']} by {query_row['artists']} ({query_row['track_genre']})")

    col_meta, col_radar = st.columns([1, 1.2])
    with col_meta:
        st.markdown("**Song Details**")
        details = {
            "Track": query_row["track_name"],
            "Artist": query_row["artists"],
            "Album": query_row["album_name"],
            "Genre": query_row["track_genre"],
            "Popularity": query_row["popularity"],
        }
        st.table(details)

    with col_radar:
        fig = build_single_radar(query_row, df_encoded, song_name=query_row["track_name"])
        st.plotly_chart(fig, use_container_width=True, key="query_radar")

    # Get recommendations based on mode
    if rec_mode == "Embedding (KNN)":
        recs, feature_comp = engine.recommend_with_features(selected_index, top_k=top_k)
    elif rec_mode == "K-Means cluster":
        km_result, _ = _fit_clusters(feature_matrix, cluster_k)
        recs = engine.recommend_by_cluster(selected_index, km_result.labels, top_k=top_k)
        feature_comp = None
        st.info(f"Recommending within K-Means cluster {int(km_result.labels[selected_index])}")
    else:  # GMM posterior
        _, gmm_result = _fit_clusters(feature_matrix, cluster_k)
        recs = engine.recommend_by_gmm(selected_index, gmm_result.probabilities, top_k=top_k)
        feature_comp = None
        st.info("Recommending by GMM posterior similarity")

    st.subheader(f"Top {len(recs)} Recommendations")
    if rerank_mode != "Default":
        st.info("Reranking is applied (demo version). This shows how results can be adjusted based on audio features.")
    st.dataframe(recs, use_container_width=True)

    # Feature comparison (build inline if not from recommend_with_features)
    if feature_comp is None:
        query_features = df_encoded.iloc[selected_index][FEATURE_COLUMNS_ENCODED].to_dict()
        query_features["track_name"] = query_row["track_name"]
        query_features["role"] = "Query"
        rows = [query_features]
        # Match rec names back to df to get feature values
        for i in range(len(recs)):
            rec_name = recs.iloc[i]["track_name"]
            rec_artist = recs.iloc[i]["artists"]
            match = df_encoded[(df_encoded["track_name"] == rec_name) & (df_encoded["artists"] == rec_artist)]
            if len(match) > 0:
                row = match.iloc[0][FEATURE_COLUMNS_ENCODED].to_dict()
                row["track_name"] = rec_name
                row["role"] = "Recommended"
                rows.append(row)
        import pandas as pd
        feature_comp = pd.DataFrame(rows)

    # Feature comparison for each recommended song
    st.subheader("Feature Comparison")
    rec_features = feature_comp[feature_comp["role"] == "Recommended"]
    for i in range(min(len(recs), len(rec_features))):
        rec_name = recs.iloc[i]["track_name"]
        rec_artist = recs.iloc[i]["artists"]
        similarity = recs.iloc[i]["similarity"]

        rec_feature_row = rec_features.iloc[i]

        with st.expander(f"{rec_name} - {rec_artist} (similarity: {similarity:.4f})"):
            explanation = explain_recommendation(
                feature_comp[feature_comp["role"] == "Query"].iloc[0],
                rec_feature_row,
            )
            st.markdown(f"*{explanation}*")

            radar_fig = build_comparison_radar(
                feature_comp[feature_comp["role"] == "Query"].iloc[0],
                rec_feature_row,
                df_encoded,
                query_name=query_row["track_name"],
                rec_name=rec_name,
            )
            st.plotly_chart(radar_fig, use_container_width=True, key=f"compare_radar_{i}")
