"""Phase 1 & 2: Song search and recommendation page with cluster-aware modes."""

from __future__ import annotations

import base64
import io
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from urllib import error, parse, request

from pathlib import Path

import joblib

from src.features import FEATURE_COLUMNS_ENCODED
from src.recommend import RecommendationEngine, rerank_mmr
from src.explain import build_comparison_radar, build_single_radar, explain_recommendation


def _plot_feature_space(
    pca_2d: np.ndarray,
    df: pd.DataFrame,
    query_idx: int,
    rec_indices: np.ndarray,
) -> go.Figure:
    """PCA-2D scatter highlighting the query song + top-K recommendations.

    Layer 1: full catalog as faint grey dots (sampled for plot performance).
    Layer 2: recommendations as numbered amber circles.
    Layer 3: query as a green star.
    """
    n = len(pca_2d)
    sample_size = min(8000, n)
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(n, size=sample_size, replace=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pca_2d[bg_idx, 0], y=pca_2d[bg_idx, 1],
        mode="markers",
        marker=dict(size=2, color="rgba(200,200,200,0.18)"),
        hoverinfo="skip", showlegend=False, name="catalog",
    ))

    # Recommendations
    if len(rec_indices) > 0:
        rec_df = df.iloc[rec_indices]
        fig.add_trace(go.Scatter(
            x=pca_2d[rec_indices, 0], y=pca_2d[rec_indices, 1],
            mode="markers+text",
            marker=dict(size=12, color="#c8956c", opacity=0.9,
                        line=dict(width=1, color="white")),
            text=[str(i + 1) for i in range(len(rec_indices))],
            textposition="top center",
            textfont=dict(size=10, color="white"),
            customdata=np.column_stack([
                rec_df["track_name"].values, rec_df["artists"].values,
                rec_df["track_genre"].values,
            ]),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                          "%{customdata[2]}<extra></extra>",
            name="Recommendations",
        ))

    # Query
    q = df.iloc[query_idx]
    fig.add_trace(go.Scatter(
        x=[pca_2d[query_idx, 0]], y=[pca_2d[query_idx, 1]],
        mode="markers",
        marker=dict(size=18, symbol="star", color="#4ade80",
                    line=dict(width=1, color="white")),
        customdata=np.array([[q["track_name"], q["artists"], q["track_genre"]]]),
        hovertemplate="<b>%{customdata[0]}</b> (query)<br>%{customdata[1]}<br>"
                      "%{customdata[2]}<extra></extra>",
        name="Query",
    ))

    fig.update_layout(
        xaxis=dict(title="PC1", gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        yaxis=dict(title="PC2", gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=40, r=20, t=40, b=40),
        height=480,
    )
    return fig


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
RECOMMEND_ARTIFACTS = [
    "feature_matrix.joblib",
    "kmeans_best.joblib",
    "gmm_full_best.joblib",
    "pca_2d.joblib",
]


# Inline coffee-gradient + music-note icon, used when Spotify cover is unavailable.
NO_COVER_DATA_URL = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 80 80'>"
    "<defs><linearGradient id='g' x1='0%25' y1='0%25' x2='100%25' y2='100%25'>"
    "<stop offset='0%25' stop-color='%23362a22'/>"
    "<stop offset='100%25' stop-color='%23211814'/>"
    "</linearGradient></defs>"
    "<rect width='80' height='80' rx='12' fill='url(%23g)'/>"
    "<path d='M52 22l-22 5v25.5a7 7 0 1 1-3-5.7V32.6l16-3.6v17.6a7 7 0 1 1-3-5.7z' "
    "fill='%23c8956c' opacity='0.85'/></svg>"
)
FALLBACK_COVER_URL = NO_COVER_DATA_URL  # back-compat alias used by older code paths


# Hand-picked cold-start suggestions (resolved to catalog indices on first load).
SUGGESTED_SONGS: list[tuple[str, str]] = [
    ("Blinding Lights", "The Weeknd"),
    ("Bohemian Rhapsody", "Queen"),
    ("bad guy", "Billie Eilish"),
    ("Smells Like Teen Spirit", "Nirvana"),
    ("Shape of You", "Ed Sheeran"),
    ("Take Five", "Dave Brubeck"),
]


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


@st.cache_data(show_spinner=False)
def _resolve_suggested_songs(_df: pd.DataFrame) -> list[dict]:
    """Resolve SUGGESTED_SONGS to actual catalog rows on first run.

    Underscore prefix on _df tells streamlit not to hash the DataFrame.
    Returns: list of {idx, track_name, artists, track_id} dicts.  Songs
    that don't match anything in the catalog are silently dropped.
    """
    out: list[dict] = []
    for name, artist in SUGGESTED_SONGS:
        m = _df[
            (_df["track_name"].str.lower().str.contains(name.lower(), na=False, regex=False))
            & (_df["artists"].str.lower().str.contains(artist.lower(), na=False, regex=False))
        ]
        if not len(m):
            continue
        idx = int(m.index[0])
        row = _df.iloc[idx]
        out.append({
            "idx": idx,
            "track_name": row["track_name"],
            "artists": row["artists"],
            "track_id": _normalize_track_id(row.get("track_id")),
        })
    return out


# Visual tokens injected once per page render.  Defined here (not in app/app.py)
# because they target structures specific to the recommendation page.
_SONG_PAGE_CSS = """
<style>
  /* Search-result + selected-song cards: lighter than page bg for layering */
  .stContainer:has(.song-row), .song-card-shell {
      background: #211814;
  }
  /* Recommendation-card text hierarchy */
  .rec-name { font-size: 16px; font-weight: 600; color: #FFFFFF; line-height: 1.2; margin: 0; }
  .rec-artist { font-size: 13px; color: #A0A0A0; margin: 2px 0 8px 0; }
  .rec-meta { font-size: 12px; color: #A0A0A0; margin-top: 6px; }
  .genre-tag {
      display: inline-block;
      background: rgba(200,149,108,0.15);
      color: #d4a87e;
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 12px;
      margin-right: 6px;
  }
  .similarity-pill {
      display: inline-block;
      font-family: ui-monospace, monospace;
      font-size: 12px;
      color: #c8956c;
  }
  /* Downgraded "Feature Comparison" expander */
  .stExpander summary p { font-size: 12px !important; color: #A0A0A0 !important; }

  /* Search row: tighter spacing */
  .song-row .stImage img { border-radius: 8px; }
</style>
"""




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
def _load_artifacts():
    """Load precomputed feature matrix and clustering artifacts."""
    missing = [n for n in RECOMMEND_ARTIFACTS if not (ARTIFACTS_DIR / n).exists()]
    if missing:
        st.error(
            "Precomputed artifacts not found: "
            + ", ".join(missing)
            + "\n\nRun `uv run python -m scripts.precompute` once to generate them."
        )
        st.stop()
    feats = joblib.load(ARTIFACTS_DIR / "feature_matrix.joblib")
    df_encoded = feats["df_encoded"]
    feature_matrix = feats["feature_matrix"]
    engine = RecommendationEngine(df_encoded, feature_matrix)
    km_result = joblib.load(ARTIFACTS_DIR / "kmeans_best.joblib")
    gmm_result = joblib.load(ARTIFACTS_DIR / "gmm_full_best.joblib")
    pca_2d = joblib.load(ARTIFACTS_DIR / "pca_2d.joblib")
    return engine, df_encoded, feature_matrix, km_result, gmm_result, pca_2d


engine, df_encoded, feature_matrix, km_result, gmm_result, pca_2d = _load_artifacts()

# Initialize page-specific session state
if "recommendation_payload" not in st.session_state:
    st.session_state["recommendation_payload"] = None
if "selected_song_index" not in st.session_state:
    st.session_state["selected_song_index"] = None

st.markdown(_SONG_PAGE_CSS, unsafe_allow_html=True)

st.title("Song Search & Recommend")
st.caption(f"{len(df_encoded):,} songs loaded from Spotify dataset")

# --- Song Search & Selection ---
search_query = st.text_input(
    "Search for a song or artist",
    placeholder="e.g. Blinding Lights, Drake, Bohemian Rhapsody...",
    key="song_search",
)

selected_index = st.session_state["selected_song_index"]


def _set_selection(idx: int) -> None:
    st.session_state["selected_song_index"] = int(idx)


# Cold start: hand-picked suggestions
if not search_query.strip():
    suggestions = _resolve_suggested_songs(engine.df)
    if suggestions:
        st.markdown("**Try one of these:**")
        chip_cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            with chip_cols[i]:
                clicked = st.button(
                    f"{sug['track_name']}\n— {sug['artists']}",
                    key=f"chip_{i}",
                    width="stretch",
                )
                if clicked:
                    _set_selection(sug["idx"])
                    st.rerun()
    if selected_index is None:
        st.info("👉 Pick one of the suggestions or type a song name above.")
        st.stop()
else:
    matches = engine.search_songs(search_query, limit=20)
    if matches.empty:
        st.warning("No matching songs found. Try a different query.")
        st.session_state["selected_song_index"] = None
        st.stop()

    # Resolve unique catalog indices in result order.
    option_indices: list[int] = []
    seen: set[int] = set()
    for _, match_row in matches.iterrows():
        found = engine.df[
            (engine.df["track_name"] == match_row["track_name"])
            & (engine.df["artists"] == match_row["artists"])
        ]
        if len(found):
            idx = int(found.index[0])
            if idx not in seen:
                seen.add(idx)
                option_indices.append(idx)

    if not option_indices:
        st.warning("Could not map search results to the recommendation catalog.")
        st.session_state["selected_song_index"] = None
        st.stop()

    if selected_index not in option_indices:
        selected_index = option_indices[0]
        st.session_state["selected_song_index"] = selected_index

    # Native selectbox — collapses automatically once a song is picked.
    selected_index = st.selectbox(
        "Pick a song from the search results",
        options=option_indices,
        index=option_indices.index(selected_index),
        format_func=lambda idx: (
            f"{engine.df.iloc[idx]['track_name']}  —  {engine.df.iloc[idx]['artists']}"
        ),
        key="search_select",
    )
    st.session_state["selected_song_index"] = int(selected_index)

# At this point we have a selected song.  Render the inline preview with album cover.
selected_index = st.session_state["selected_song_index"]
selected_row = engine.df.iloc[int(selected_index)]
selected_track_id = _normalize_track_id(selected_row.get("track_id"))
selected_release_date = ""
selected_album_cover_url = NO_COVER_DATA_URL
if selected_track_id:
    cid, sec = _get_spotify_credentials()
    if cid and sec:
        meta = _get_spotify_track_metadata(selected_track_id, cid, sec)
        selected_release_date = meta.get("spotify_release_date", "")
        selected_album_cover_url = meta.get("album_cover_url", "") or NO_COVER_DATA_URL

with st.container(border=True):
    card_left, card_right = st.columns([1, 4])
    with card_left:
        st.image(selected_album_cover_url, width=110)
    with card_right:
        st.markdown(
            f"<div class='rec-name'>{selected_row['track_name']}</div>"
            f"<div class='rec-artist'>{selected_row['artists']}</div>"
            f"<div class='rec-meta'>"
            f"<span class='genre-tag'>{selected_row.get('track_genre', '—')}</span>"
            f"Album: {selected_row.get('album_name', '—')} · "
            f"Pop {int(selected_row.get('popularity', 0))}"
            f"{' · ' + selected_release_date if selected_release_date else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    seed_action_col1, seed_action_col2, _ = st.columns([1, 1, 4])
    add_to_player = seed_action_col1.button(
        "Add to Player",
        key="add_seed_to_player",
        disabled=selected_track_id is None,
    )
    if selected_track_id:
        spotify_track_url, _spotify_embed = _build_spotify_urls(selected_track_id)
        seed_action_col2.markdown(f"[🎵 Open in Spotify]({spotify_track_url})")
    else:
        seed_action_col2.caption("❌ Spotify link unavailable.")
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

# Recommendation mode
rec_mode = st.radio(
    "Recommendation mode",
    options=["K-Means cluster", "GMM posterior"],
    horizontal=True,
)

st.subheader("Diversity Reranking (MMR)")
st.caption(
    "Maximum Marginal Relevance reranks the candidate pool by trading off "
    "similarity to the query against diversity within the recommendations. "
    "Higher λ = closer to query; lower λ = more variety."
)

rerank_mode = st.radio(
    "Reranking",
    ["Off (relevance only)", "MMR (diverse)"],
    horizontal=True,
)
mmr_lambda = st.select_slider(
    "λ (relevance ↔ diversity)",
    options=[0.5, 0.7, 0.9],
    value=0.7,
    disabled=(rerank_mode != "MMR (diverse)"),
    key="mmr_lambda",
)

top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=10)

if rec_mode in ("K-Means cluster", "GMM posterior"):
    _k_used = km_result.n_clusters if rec_mode == "K-Means cluster" else gmm_result.n_clusters
    st.caption(f"Using precomputed model with K={_k_used}.")

if st.button("Recommend", type="primary"):
    # Magical effect: spinner while computing
    with st.spinner("✨ Analyzing audio dimensions... 🎵"):
        # If MMR is on we ask for an over-fetched candidate pool so MMR has room
        # to diversify; otherwise we just take the requested top_k directly.
        use_mmr = (rerank_mode == "MMR (diverse)")
        fetch_k = max(top_k * 5, 50) if use_mmr else top_k

        cluster_message = None
        feature_comp = None
        if rec_mode == "K-Means cluster":
            recs = engine.recommend_by_cluster(selected_index, km_result.labels, top_k=fetch_k)
            cluster_message = f"Recommending within K-Means cluster {int(km_result.labels[selected_index])}"
        else:  # GMM posterior
            recs = engine.recommend_by_gmm(selected_index, gmm_result.probabilities, top_k=fetch_k)
            cluster_message = "Recommending by GMM posterior similarity"

        if use_mmr:
            recs = rerank_mmr(engine, recs, selected_index, lam=float(mmr_lambda), top_k=top_k)

        recs = _attach_spotify_fields(recs)
        recs = _attach_spotify_metadata(recs)

        st.session_state["recommendation_payload"] = {
            "selected_index": int(selected_index),
            "rec_mode": rec_mode,
            "rerank_mode": rerank_mode,
            "mmr_lambda": float(mmr_lambda),
            "cluster_message": cluster_message,
            "recs": recs,
            "feature_comp": feature_comp,
        }
    
    # Celebrate the results! 🎉
    st.balloons()

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

    with col_radar:
        fig = build_single_radar(query_row, df_encoded, song_name=query_row["track_name"])
        st.plotly_chart(fig, width='stretch', key="query_radar")

    if payload["cluster_message"]:
        st.info(payload["cluster_message"])

    st.subheader(f"Top {len(recs)} Recommendations")
    
    # ✨ Magical animated styles for results
    st.markdown(
        """
        <style>
        @keyframes aiGlow {
            0% { box-shadow: 0 0 5px rgba(200, 149, 108, 0.3); }
            50% { box-shadow: 0 0 20px rgba(200, 149, 108, 0.6), 0 0 40px rgba(74, 222, 128, 0.2); }
            100% { box-shadow: 0 0 5px rgba(200, 149, 108, 0.3); }
        }
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        [data-testid="stContainer"] > [data-testid="stVerticalBlock"]:has([data-testid="stDownloadButton"]) ~ [data-testid="stVerticalBlock"] {
            animation: slideInLeft 0.6s ease-out;
        }
        div[data-testid="stContainer"] {
            animation: aiGlow 3s ease-in-out infinite;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Add export button for recommendations
    if len(recs) > 0:
        first_song = recs.iloc[0]["track_name"].replace(" ", "_").replace("/", "_")
        csv_buffer = io.StringIO()
        recs.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        st.download_button(
            label="📥 Download Recommendations as CSV",
            data=csv_data,
            file_name=f"{first_song}_recommendations.csv",
            mime="text/csv",
            key="download_recommendations",
        )
    
    if payload["rerank_mode"] == "MMR (diverse)":
        st.info(f"Reranked with MMR (λ={payload['mmr_lambda']}).")

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
            card_left, card_right = st.columns([1, 4])
            with card_left:
                cover_url = rec.get("album_cover_url", "") or NO_COVER_DATA_URL
                st.image(cover_url, width=110)
            with card_right:
                release_date = rec.get("spotify_release_date", "")
                meta_bits = [
                    f"<span class='genre-tag'>{rec_genre}</span>",
                    f"<span class='similarity-pill'>Similarity {similarity:.3f}</span>",
                ]
                meta_line2 = (
                    f"Album: {rec_album} · Pop {rec_popularity}"
                    + (f" · {release_date}" if release_date else "")
                )
                st.markdown(
                    f"<div class='rec-name'>{i + 1}. {rec_name}</div>"
                    f"<div class='rec-artist'>{rec_artist}</div>"
                    f"<div>{''.join(meta_bits)}</div>"
                    f"<div class='rec-meta'>{meta_line2}</div>",
                    unsafe_allow_html=True,
                )

                action_col1, action_col2 = st.columns([1, 2])
                if action_col1.button(
                    "Add to Player",
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
                    action_col2.caption("❌ Spotify link unavailable.")

            rec_feature_match = rec_feature_rows[
                (rec_feature_rows["track_name"] == rec_name)
                & (rec_feature_rows["artists"] == rec_artist)
            ]
            if len(rec_feature_match) > 0:
                rec_feature_row = rec_feature_match.iloc[0]
                with st.expander("Feature comparison", expanded=False):
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

    # ── Feature Space (PCA) panel ────────────────────────────────────────────
    st.subheader("Feature Space (PCA)")
    st.caption(
        "Catalog projected to 2-D via PCA.  The query song (★) and top-K "
        "recommendations (numbered amber circles) are highlighted, so you can "
        "see where they live relative to the rest of the catalog."
    )
    rec_indices: list[int] = []
    for _, rec in recs.iterrows():
        match = engine.df[
            (engine.df["track_name"] == rec["track_name"])
            & (engine.df["artists"] == rec["artists"])
        ]
        if len(match):
            rec_indices.append(int(match.index[0]))
    pca_fig = _plot_feature_space(
        pca_2d, engine.df, selected_index, np.asarray(rec_indices, dtype=int),
    )
    st.plotly_chart(pca_fig, width="stretch", key="recommend_pca")
