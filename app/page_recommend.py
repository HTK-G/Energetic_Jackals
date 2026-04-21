"""Phase 1 & 2: Song search and recommendation page with cluster-aware modes."""

from __future__ import annotations

import base64
import json
import pandas as pd
import streamlit as st
from urllib import error, parse, request

from src.clustering import fit_gmm, fit_kmeans
from src.features import FEATURE_COLUMNS_ENCODED, build_feature_matrix, load_dataset
from src.recommend import RecommendationEngine
from src.explain import build_comparison_radar, build_single_radar, explain_recommendation


FALLBACK_COVER_URL = "https://tenor.com/view/jenminismo-gif-24558731"


def _normalize_track_id(value: object) -> str | None:
    """Return a cleaned Spotify track_id string or None if invalid."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _build_spotify_urls(track_id: str) -> tuple[str, str]:
    """Build standard Spotify links from a valid track_id."""
    return (
        f"https://open.spotify.com/track/{track_id}",
        f"https://open.spotify.com/embed/track/{track_id}?autoplay=1",
    )


def _set_active_player(track_name: str, artists: str, embed_url: str) -> None:
    """Set the active Spotify player state for sidebar rendering."""
    st.session_state["active_spotify_embed_url"] = embed_url
    st.session_state["active_spotify_label"] = f"Now playing: {track_name} - {artists}"


def _seed_song_label(df: pd.DataFrame, index: int) -> str:
    """Standardized seed-song label with rich context for the selectbox."""
    row = df.iloc[index]
    return (
        f"{row['track_name']} | {row['artists']} | "
        f"{row.get('album_name', 'Unknown Album')} | "
        f"{row['track_genre']} | Pop {int(row['popularity'])}"
    )


def _get_spotify_credentials() -> tuple[str | None, str | None]:
    """Read Spotify API credentials from Streamlit secrets if configured."""
    try:
        client_id = st.secrets.get("SPOTIFY_CLIENT_ID")
        client_secret = st.secrets.get("SPOTIFY_CLIENT_SECRET")
    except Exception:
        return None, None

    if not client_id or not client_secret:
        return None, None
    return str(client_id), str(client_secret)


@st.cache_data(ttl=3300, show_spinner=False)
def _get_spotify_access_token(client_id: str, client_secret: str) -> str | None:
    """Retrieve an app access token using Spotify Client Credentials flow."""
    token_url = "https://accounts.spotify.com/api/token"
    encoded = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
    data = parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
    headers = {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    req = request.Request(token_url, data=data, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    return payload.get("access_token")


@st.cache_data(ttl=86400, show_spinner=False)
def _get_spotify_track_metadata(track_id: str, client_id: str, client_secret: str) -> dict:
    """Fetch Spotify track metadata (album image/details) by track id."""
    token = _get_spotify_access_token(client_id, client_secret)
    if not token:
        return {}

    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    headers = {"Authorization": f"Bearer {token}"}
    req = request.Request(url, headers=headers, method="GET")

    try:
        with request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
        return {}

    album = payload.get("album") or {}
    images = album.get("images") or []
    cover_url = images[0].get("url") if images else ""

    return {
        "album_cover_url": cover_url,
        "spotify_album_name": album.get("name", ""),
        "spotify_release_date": album.get("release_date", ""),
    }


def _attach_spotify_metadata(recs: pd.DataFrame) -> pd.DataFrame:
    """Attach album-cover metadata from Spotify API when credentials are present."""
    out = recs.copy()
    out["album_cover_url"] = ""
    out["spotify_album_name"] = ""
    out["spotify_release_date"] = ""

    client_id, client_secret = _get_spotify_credentials()
    if not client_id or not client_secret:
        return out

    for i in range(len(out)):
        track_id = _normalize_track_id(out.iloc[i].get("track_id"))
        if not track_id:
            continue
        meta = _get_spotify_track_metadata(track_id, client_id, client_secret)
        if not meta:
            continue
        out.at[i, "album_cover_url"] = meta.get("album_cover_url", "")
        out.at[i, "spotify_album_name"] = meta.get("spotify_album_name", "")
        out.at[i, "spotify_release_date"] = meta.get("spotify_release_date", "")

    return out


def _attach_spotify_fields(recs: pd.DataFrame) -> pd.DataFrame:
    """Attach spotify availability and URLs after recommendation output is ready."""
    out = recs.copy()
    if "track_id" not in out.columns:
        out["track_id"] = None

    out["track_id"] = out["track_id"].apply(_normalize_track_id)
    out["spotify_available"] = out["track_id"].notna()
    out["spotify_url"] = out["track_id"].apply(
        lambda tid: _build_spotify_urls(tid)[0] if tid else ""
    )
    out["spotify_embed_url"] = out["track_id"].apply(
        lambda tid: _build_spotify_urls(tid)[1] if tid else ""
    )
    return out


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

if "active_spotify_embed_url" not in st.session_state:
    st.session_state["active_spotify_embed_url"] = None
if "active_spotify_label" not in st.session_state:
    st.session_state["active_spotify_label"] = ""
if "active_spotify_meta" not in st.session_state:
    st.session_state["active_spotify_meta"] = None
if "recommendation_payload" not in st.session_state:
    st.session_state["recommendation_payload"] = None
if "selected_seed_index" not in st.session_state:
    st.session_state["selected_seed_index"] = None

with st.sidebar:
    st.subheader("Spotify Player")
    if st.session_state["active_spotify_embed_url"]:
        st.caption(st.session_state["active_spotify_label"])
        meta = st.session_state.get("active_spotify_meta")
        if isinstance(meta, dict):
            cover_url = meta.get("album_cover_url", "")
            st.caption(
                f"{meta.get('album_name', 'Unknown Album')} | "
                f"{meta.get('track_genre', 'Unknown Genre')} | "
                f"Popularity {meta.get('popularity', 'N/A')}"
            )
            if cover_url:
                st.image(cover_url, width=240)
        st.components.v1.iframe(
            st.session_state["active_spotify_embed_url"],
            height=380,
            scrolling=False,
        )
    else:
        st.caption("No song selected yet. Use Play or Add to Player.")

st.title("Song Search & Recommend")
st.caption(f"{len(df_encoded):,} songs loaded from Spotify dataset")

# --- Song Search & Selection ---
st.subheader("Find Your Seed Song")
st.caption("Search for a song or artist and select it to find similar recommendations.")

search_query = st.text_input(
    "Search for a song or artist",
    placeholder="e.g. Blinding Lights, Drake, Bohemian Rhapsody...",
    key="seed_search",
)

selected_index = st.session_state["selected_seed_index"]

if search_query.strip():
    matches = engine.search_songs(search_query, limit=50)
    if matches.empty:
        st.warning("No matching songs found. Try a different query.")
        st.session_state["selected_seed_index"] = None
        selected_index = None
    else:
        # Map matches to catalog indices and keep only unique indices.
        option_indices: list[int] = []
        seen_indices: set[int] = set()
        for _, match_row in matches.iterrows():
            found = engine.df[
                (engine.df["track_name"] == match_row["track_name"]) &
                (engine.df["artists"] == match_row["artists"])
            ]
            if len(found) > 0:
                idx = int(found.index[0])
                if idx not in seen_indices:
                    option_indices.append(idx)
                    seen_indices.add(idx)

        if not option_indices:
            st.warning("Could not map search results to the recommendation catalog.")
            st.session_state["selected_seed_index"] = None
            selected_index = None
        else:
            if selected_index not in option_indices:
                selected_index = option_indices[0]
                st.session_state["selected_seed_index"] = selected_index

            selected_index = st.selectbox(
                "Select a song to use as your seed",
                options=option_indices,
                index=option_indices.index(selected_index),
                format_func=lambda idx: _seed_song_label(engine.df, idx),
                key="search_select",
            )

            selected_row = engine.df.iloc[int(selected_index)]
            selected_track_id = _normalize_track_id(selected_row.get("track_id"))
            selected_cover_url = ""
            selected_release_date = ""
            client_id, client_secret = _get_spotify_credentials()
            if selected_track_id and client_id and client_secret:
                selected_meta = _get_spotify_track_metadata(selected_track_id, client_id, client_secret)
                selected_cover_url = selected_meta.get("album_cover_url", "")
                selected_release_date = selected_meta.get("spotify_release_date", "")

            with st.container(border=True):
                seed_left, seed_right = st.columns([1, 3])
                with seed_left:
                    st.image(selected_cover_url if selected_cover_url else FALLBACK_COVER_URL, width=140)
                with seed_right:
                    st.markdown(f"**{selected_row['track_name']}**")
                    st.write(
                        f"{selected_row['artists']} | Album: {selected_row.get('album_name', 'Unknown Album')} | "
                        f"Genre: {selected_row.get('track_genre', 'Unknown Genre')} | "
                        f"Popularity: {int(selected_row.get('popularity', 0))}"
                    )
                    if selected_release_date:
                        st.caption(f"Release Date: {selected_release_date}")

                seed_action_col1, seed_action_col2 = st.columns([1, 2])
                add_to_player = seed_action_col1.button(
                    "Add to Player",
                    key="add_seed_to_player",
                    disabled=selected_track_id is None,
                )
                if selected_track_id:
                    spotify_track_url, _ = _build_spotify_urls(selected_track_id)
                    seed_action_col2.markdown(f"[🎵 Open in Spotify]({spotify_track_url})")
                else:
                    seed_action_col2.caption("❌ Spotify link unavailable for this seed song.")

            if add_to_player and selected_track_id is not None:
                _, selected_embed_url = _build_spotify_urls(selected_track_id)
                _set_active_player(
                    selected_row["track_name"],
                    selected_row["artists"],
                    selected_embed_url,
                )
                st.session_state["active_spotify_meta"] = {
                    "album_name": selected_row.get("album_name", "Unknown Album"),
                    "track_genre": selected_row.get("track_genre", "Unknown Genre"),
                    "popularity": int(selected_row.get("popularity", 0)),
                    "album_cover_url": "",
                }
                st.rerun()

            st.session_state["selected_seed_index"] = int(selected_index)
else:
    st.info("👉 Start by searching for a song or artist to select it as your seed song.")
    st.session_state["selected_seed_index"] = None
    selected_index = None

if selected_index is None:
    st.stop()

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
    # Get recommendations based on mode
    cluster_message = None
    if rec_mode == "Embedding (KNN)":
        recs, feature_comp = engine.recommend_with_features(selected_index, top_k=top_k)
    elif rec_mode == "K-Means cluster":
        km_result, _ = _fit_clusters(feature_matrix, cluster_k)
        recs = engine.recommend_by_cluster(selected_index, km_result.labels, top_k=top_k)
        feature_comp = None
        cluster_message = f"Recommending within K-Means cluster {int(km_result.labels[selected_index])}"
    else:  # GMM posterior
        _, gmm_result = _fit_clusters(feature_matrix, cluster_k)
        recs = engine.recommend_by_gmm(selected_index, gmm_result.probabilities, top_k=top_k)
        feature_comp = None
        cluster_message = "Recommending by GMM posterior similarity"

    recs = _attach_spotify_fields(recs)
    recs = _attach_spotify_metadata(recs)

    st.session_state["recommendation_payload"] = {
        "selected_index": int(selected_index),
        "rec_mode": rec_mode,
        "rerank_mode": rerank_mode,
        "cluster_message": cluster_message,
        "recs": recs,
        "feature_comp": feature_comp,
    }

payload = st.session_state["recommendation_payload"]
if payload is not None:
    selected_index = int(payload["selected_index"])
    recs = payload["recs"]
    feature_comp = payload["feature_comp"]
    query_row = engine.df.iloc[selected_index]

    st.markdown(
        f"**Query song**: {query_row['track_name']} by {query_row['artists']} ({query_row['track_genre']})"
    )

    col_meta, col_radar = st.columns([1, 1.2])
    with col_meta:
        st.markdown("**Song Details**")
        details_rows = [
            ("Track", query_row["track_name"]),
            ("Artist", query_row["artists"]),
            ("Album", query_row["album_name"]),
            ("Genre", query_row["track_genre"]),
            ("Popularity", query_row["popularity"]),
        ]
        for label, value in details_rows:
            st.write(f"**{label}:** {value}")
        
        # Add Spotify player controls for query song
        query_track_id = _normalize_track_id(query_row.get("track_id"))
        query_spotify_available = query_track_id is not None
        
        if query_spotify_available:
            query_spotify_url, query_spotify_embed_url = _build_spotify_urls(query_track_id)
            st.divider()
            col_play, col_open = st.columns([1, 2])
            
            if col_play.button("▶ Play Query Song", key="play_query_song"):
                _set_active_player(
                    query_row["track_name"],
                    query_row["artists"],
                    query_spotify_embed_url,
                )
                st.session_state["active_spotify_meta"] = {
                    "album_name": query_row.get("album_name", "Unknown Album"),
                    "track_genre": query_row.get("track_genre", "Unknown Genre"),
                    "popularity": int(query_row.get("popularity", 0)),
                    "album_cover_url": "",
                }
                st.rerun()
            
            col_open.markdown(f"[🎵 Open in Spotify]({query_spotify_url})")
        else:
            st.caption("❌ Spotify ID not available for this song.")

    with col_radar:
        fig = build_single_radar(query_row, df_encoded, song_name=query_row["track_name"])
        st.plotly_chart(fig, width='stretch', key="query_radar")

    if payload["cluster_message"]:
        st.info(payload["cluster_message"])

    st.subheader(f"Top {len(recs)} Recommendations")
    if payload["rerank_mode"] != "Default":
        st.info("Reranking is applied (demo version). This shows how results can be adjusted based on audio features.")

    # Feature comparison (build inline if not from recommend_with_features)
    if feature_comp is None:
        query_features = df_encoded.iloc[selected_index][FEATURE_COLUMNS_ENCODED].to_dict()
        query_features["track_name"] = query_row["track_name"]
        query_features["artists"] = query_row["artists"]
        query_features["role"] = "Query"
        rows = [query_features]
        for i in range(len(recs)):
            rec_name = recs.iloc[i]["track_name"]
            rec_artist = recs.iloc[i]["artists"]
            match = df_encoded[(df_encoded["track_name"] == rec_name) & (df_encoded["artists"] == rec_artist)]
            if len(match) > 0:
                row = match.iloc[0][FEATURE_COLUMNS_ENCODED].to_dict()
                row["track_name"] = rec_name
                row["artists"] = rec_artist
                row["role"] = "Recommended"
                rows.append(row)
        feature_comp = pd.DataFrame(rows)

    query_feature_row = feature_comp[feature_comp["role"] == "Query"].iloc[0]
    rec_feature_rows = feature_comp[feature_comp["role"] == "Recommended"]

    for i, rec in recs.reset_index(drop=True).iterrows():
        rec_name = rec["track_name"]
        rec_artist = rec["artists"]
        rec_album = rec.get("album_name", "Unknown Album")
        rec_genre = rec["track_genre"]
        rec_popularity = int(rec["popularity"])
        similarity = float(rec["similarity"])
        spotify_available = bool(rec["spotify_available"])

        with st.container(border=True):
            card_left, card_right = st.columns([1, 3])
            with card_left:
                cover_url = rec.get("album_cover_url", "")
                st.image(cover_url if cover_url else FALLBACK_COVER_URL, width=140)
            with card_right:
                st.markdown(f"**{rec_name}**")
                st.write(
                    f"{rec_artist} | Album: {rec_album} | Genre: {rec_genre} | "
                    f"Popularity: {rec_popularity} | Similarity: {similarity:.4f}"
                )
                release_date = rec.get("spotify_release_date", "")
                if release_date:
                    st.caption(f"Release Date: {release_date}")

            action_col1, action_col2 = st.columns([1, 2])
            if action_col1.button(
                "▶ Play",
                key=f"play_{selected_index}_{i}",
                disabled=not spotify_available,
            ):
                _set_active_player(rec_name, rec_artist, rec["spotify_embed_url"])
                st.session_state["active_spotify_meta"] = {
                    "album_name": rec_album,
                    "track_genre": rec_genre,
                    "popularity": rec_popularity,
                    "album_cover_url": rec.get("album_cover_url", ""),
                }
                st.rerun()

            if spotify_available:
                action_col2.markdown(f"[🎵 Open in Spotify]({rec['spotify_url']})")
            else:
                action_col2.caption("❌ Spotify link unavailable for this recommendation.")

            rec_feature_match = rec_feature_rows[
                (rec_feature_rows["track_name"] == rec_name)
                & (rec_feature_rows["artists"] == rec_artist)
            ]
            if len(rec_feature_match) > 0:
                rec_feature_row = rec_feature_match.iloc[0]
                with st.expander("Feature Comparison", expanded=False):
                    explanation = explain_recommendation(query_feature_row, rec_feature_row)
                    st.markdown(f"*{explanation}*")

                    radar_fig = build_comparison_radar(
                        query_feature_row,
                        rec_feature_row,
                        df_encoded,
                        query_name=query_row["track_name"],
                        rec_name=rec_name,
                    )
                    st.plotly_chart(radar_fig, width="stretch", key=f"compare_radar_{i}")
