from __future__ import annotations



import pandas as pd
import streamlit as st

from src.features import FEATURE_COLUMNS_ENCODED, build_feature_matrix, load_dataset
from src.recommend import RecommendationEngine
# from src.explain import build_single_radar
from src.explain import build_single_radar, build_comparison_radar, explain_recommendation


@st.cache_resource
def _build_engine():
    """Load data, build features, and create the recommendation engine."""
    df = load_dataset()
    feature_matrix, scaler, df_encoded = build_feature_matrix(df)
    engine = RecommendationEngine(df_encoded, feature_matrix)
    return engine, df_encoded


engine, df_encoded = _build_engine()

st.title("Playlist-Based Recommendation")
st.caption("Select multiple songs and get recommendations from their combined audio profile.")

# ---------- Session state ----------
if "playlist_seed_indices" not in st.session_state:
    st.session_state.playlist_seed_indices = []

# ---------- Search ----------
search_query = st.text_input(
    "Search for songs to add to the playlist",
    placeholder="e.g. Drake, Blinding Lights, Hello",
)

search_results = None
selected_search_indices = []

if search_query.strip():
    matches = engine.search_songs(search_query, limit=20)

    if matches.empty:
        st.warning("No matching songs found. Try a different query.")
    else:
        st.subheader("Search Results")
        st.dataframe(matches.reset_index(drop=True), use_container_width=True)
        search_results = matches

        search_option_labels = [
            f"{row['track_name']} - {row['artists']} ({row['album_name']})"
            for _, row in search_results.iterrows()
        ]

        selected_labels = st.multiselect(
            "Pick songs from search results",
            options=search_option_labels,
            key="playlist_search_multi_select",
        )

        for label in selected_labels:
            selected_row = search_results[
                (
                    search_results["track_name"]
                    + " - "
                    + search_results["artists"]
                    + " ("
                    + search_results["album_name"]
                    + ")"
                ) == label
            ].iloc[0]

            match_df = engine.df[
                (engine.df["track_name"] == selected_row["track_name"])
                & (engine.df["artists"] == selected_row["artists"])
                & (engine.df["album_name"] == selected_row["album_name"])
            ]

            if len(match_df) > 0:
                selected_search_indices.append(match_df.index[0])

        if st.button("Add selected songs to playlist", type="secondary"):
            added_count = 0
            for idx in selected_search_indices:
                if idx not in st.session_state.playlist_seed_indices:
                    st.session_state.playlist_seed_indices.append(idx)
                    added_count += 1

            if added_count > 0:
                st.success(f"Added {added_count} song(s) to the playlist seeds.")
            else:
                st.info("No new songs were added.")

st.divider()
st.subheader("Selected Playlist Seeds")

playlist_seed_indices = st.session_state.playlist_seed_indices

if len(playlist_seed_indices) == 0:
    st.info("No songs selected yet.")
else:
    for idx in playlist_seed_indices:
        row = engine.df.iloc[idx]
        st.markdown(f"- {row['track_name']} by {row['artists']}")

    if st.button("Clear playlist seeds"):
        st.session_state.playlist_seed_indices = []
        st.rerun()

# ---------- Playlist profile ----------
if len(playlist_seed_indices) > 0:
    st.divider()

    playlist_rows = df_encoded.iloc[playlist_seed_indices]
    avg_features = playlist_rows[FEATURE_COLUMNS_ENCODED].mean()

    profile_df = pd.DataFrame({
        "feature": avg_features.index,
        "average_value": avg_features.values,
    })

    col_profile, col_radar = st.columns([1, 1.2])

    with col_profile:
        st.subheader("Playlist Seed Profile")
        st.caption("Average audio-feature values across the selected playlist seeds.")
        st.dataframe(profile_df, use_container_width=True)

    with col_radar:
        st.subheader("Playlist Feature Radar")
        avg_row = avg_features.copy()
        fig = build_single_radar(
            avg_row,
            df_encoded,
            song_name="Playlist Profile"
        )
        st.plotly_chart(fig, use_container_width=True)



top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=10)

# ---------- Recommendation ----------
if st.button("Generate Playlist Recommendations", type="primary"):
    if len(playlist_seed_indices) == 0:
        st.warning("Please add at least one song to the playlist seeds.")
        st.stop()

    recs = engine.recommend_from_playlist(playlist_seed_indices, top_k=top_k)

    st.subheader(f"Top {len(recs)} Recommendations")
    st.info("Recommendations are based on the average audio-feature profile of the selected songs.")
    st.dataframe(recs, use_container_width=True)

    # ----- Build playlist average query row -----
    playlist_rows = df_encoded.iloc[playlist_seed_indices]
    avg_features = playlist_rows[FEATURE_COLUMNS_ENCODED].mean()

    query_features = avg_features.to_dict()
    query_features["track_name"] = "Playlist Profile"
    query_features["role"] = "Query"

    rows = [query_features]

    # Match recommended songs back to df_encoded to get their feature rows
    for i in range(len(recs)):
        rec_name = recs.iloc[i]["track_name"]
        rec_artist = recs.iloc[i]["artists"]

        match = df_encoded[
            (df_encoded["track_name"] == rec_name) &
            (df_encoded["artists"] == rec_artist)
        ]

        if len(match) > 0:
            row = match.iloc[0][FEATURE_COLUMNS_ENCODED].to_dict()
            row["track_name"] = rec_name
            row["role"] = "Recommended"
            rows.append(row)

    feature_comp = pd.DataFrame(rows)

    # ----- Feature Comparison -----
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
            # st.markdown(f"*{explanation}*")
            st.markdown("**Why this song was recommended**")
            st.write(explanation)
            st.markdown("---")

            radar_fig = build_comparison_radar(
                feature_comp[feature_comp["role"] == "Query"].iloc[0],
                rec_feature_row,
                df_encoded,
                query_name="Playlist Profile",
                rec_name=rec_name,
            )

            st.plotly_chart(
                radar_fig,
                use_container_width=True,
                key=f"playlist_compare_radar_{i}",
            )