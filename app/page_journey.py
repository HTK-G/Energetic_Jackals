"""Music Journey — trajectory playlists and GMM-cluster endless radio.

Tab 1: Scenario Presets — 6 data-driven scenarios with trajectory playlists
Tab 2: Custom Mode — user-defined start/end via sliders → trajectory playlist
Tab 3: Endless Radio — GMM-cluster roaming with Bayesian belief update
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from urllib import error, parse, request

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.journey import (
    TRAJECTORY_FEATURES,
    build_journey_playlist,
    build_trajectory_features,
    gmm_endless_next,
)
from src.visualization import plot_endless_history, plot_journey

# ── Spotify player utilities ─────────────────────────────────────────────────

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


# Visual tokens for song cards
_SONG_JOURNEY_CSS = """
<style>
  /* Song card styling */
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
</style>
"""

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

SCENARIO_DISPLAY = {
    "workout": ("🏋️", "Workout", "Low → high energy & tempo ramp"),
    "focus": ("📚", "Focus", "Calm ambient to light concentration flow"),
    "wind down": ("🌙", "Wind Down", "Gradual descent into relaxation"),
    "party": ("🎉", "Party", "Build-up from warm-up to peak energy"),
    "commute": ("🚗", "Commute", "Steady-state easy listening"),
    "rainy night": ("🌧️", "Rainy Night", "Mellow, jazzy, introspective"),
}


@st.cache_resource
def _load_journey_data():
    """Load feature matrix, scenario mappings, and GMM model."""
    fm_path = ARTIFACTS_DIR / "feature_matrix.joblib"
    sc_path = ARTIFACTS_DIR / "scenario_mappings.joblib"
    gmm_path = ARTIFACTS_DIR / "gmm_full_best.joblib"
    missing = [str(p) for p in [fm_path, sc_path, gmm_path] if not p.exists()]
    if missing:
        st.error(
            "Missing artifacts: " + ", ".join(missing)
            + "\n\nRun `uv run python -m scripts.precompute` and "
            "`uv run python -m scripts.derive_scenario_mappings` first."
        )
        st.stop()
    feats = joblib.load(fm_path)
    df_encoded = feats["df_encoded"]
    traj_matrix = build_trajectory_features(df_encoded)
    scenarios = joblib.load(sc_path)
    gmm = joblib.load(gmm_path)
    return df_encoded, traj_matrix, scenarios, gmm


df_encoded, traj_matrix, scenarios, gmm_result = _load_journey_data()

st.markdown(_SONG_JOURNEY_CSS, unsafe_allow_html=True)

st.title("Music Journey")
st.caption("Trajectory playlists and GMM-cluster roaming in Russell emotion space")

tab_scenario, tab_custom, tab_endless = st.tabs([
    "🎬 Scenario Presets", "🎛️ Custom Trajectory", "📻 Endless Radio (GMM)"
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Scenario Presets
# ══════════════════════════════════════════════════════════════════════════════

with tab_scenario:
    st.subheader("Choose a Scenario")
    st.caption("Each scenario uses data-driven start/end points derived from genre-matched songs.")

    # Scenario cards in a 3-column grid
    cols = st.columns(3)
    selected_scenario = st.session_state.get("selected_scenario")

    for i, (key, (emoji, label, desc)) in enumerate(SCENARIO_DISPLAY.items()):
        col = cols[i % 3]
        with col:
            with st.container(border=True):
                st.markdown(f"### {emoji} {label}")
                st.caption(desc)
                n = scenarios[key]["n_songs"]
                st.caption(f"Based on {n:,} matched songs")
                if st.button(f"Generate {label} Playlist", key=f"btn_{key}",
                             type="primary" if selected_scenario == key else "secondary"):
                    st.session_state["selected_scenario"] = key
                    st.session_state.pop("scenario_playlist", None)
                    st.rerun()

    # Generate and display playlist
    if selected_scenario and selected_scenario in scenarios:
        mapping = scenarios[selected_scenario]
        start = mapping["start"]
        end = mapping["end"]

        # Handle scenarios where name implies descending but data is ascending
        if selected_scenario in ("wind down", "rainy night"):
            start, end = end, start  # swap: high → low

        n_songs = st.slider("Number of songs", 5, 15, 10, key="scenario_n")
        pop_w = st.slider("Popularity weight", 0.0, 1.0, 0.5, 0.1,
                          key="scenario_pop",
                          help="Higher = prefer popular songs; 0 = purely feature-based")

        # ±0.05 uniform jitter on start + end so each Regenerate gives a
        # different sequence while staying near the scenario's audio character.
        SCENARIO_JITTER = 0.05

        def _generate_scenario_playlist() -> pd.DataFrame:
            return build_journey_playlist(
                start, end, traj_matrix, df_encoded, n_songs=n_songs,
                popularity_weight=pop_w, jitter=SCENARIO_JITTER,
                rng=np.random.default_rng(),
            )

        if "scenario_playlist" not in st.session_state:
            st.session_state["scenario_playlist"] = _generate_scenario_playlist()
        playlist = st.session_state["scenario_playlist"]

        if st.button("🔄 Regenerate", key="regen_scenario"):
            st.session_state["scenario_playlist"] = _generate_scenario_playlist()
            st.rerun()

        # Visualization
        emoji, label, _ = SCENARIO_DISPLAY[selected_scenario]
        st.subheader(f"{emoji} {label} — Russell Emotion Space")
        from src.journey import generate_trajectory
        waypoints = generate_trajectory(start, end, n_songs)
        fig = plot_journey(waypoints, playlist, df_encoded, start, end)
        st.plotly_chart(fig, width="stretch", key="scenario_plot")

        # Export playlist button
        st.subheader("Playlist")
        # ✨ Magical animated styles
        st.markdown(
            """
            <style>
            @keyframes aiGlow {
                0% { box-shadow: 0 0 5px rgba(200, 149, 108, 0.3); }
                50% { box-shadow: 0 0 20px rgba(200, 149, 108, 0.6), 0 0 40px rgba(74, 222, 128, 0.2); }
                100% { box-shadow: 0 0 5px rgba(200, 149, 108, 0.3); }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if len(playlist) > 0:
            first_song = playlist.iloc[0]["track_name"].replace(" ", "_").replace("/", "_")
            csv_buffer = io.StringIO()
            playlist.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            st.download_button(
                label="📥 Download Playlist as CSV",
                data=csv_data,
                file_name=f"{first_song}_recommendations.csv",
                mime="text/csv",
                key=f"download_scenario_{selected_scenario}",
            )
        for idx, (_, row) in enumerate(playlist.iterrows()):
            step = int(row["step"]) + 1
            track_id = _normalize_track_id(row.get("track_id"))
            spotify_available = track_id is not None
            
            # Fetch album cover if Spotify credentials are available
            album_cover_url = NO_COVER_DATA_URL
            if spotify_available:
                client_id, client_secret = _get_spotify_credentials()
                if client_id and client_secret:
                    meta = _get_spotify_track_metadata(track_id, client_id, client_secret)
                    album_cover_url = meta.get("album_cover_url", "") or NO_COVER_DATA_URL
            
            with st.container(border=True):
                card_left, card_right = st.columns([1, 4])
                with card_left:
                    st.image(album_cover_url, width=110)
                with card_right:
                    st.markdown(
                        f"<div class='rec-name'>{step}. {row['track_name']}</div>"
                        f"<div class='rec-artist'>{row['artists']}</div>"
                        f"<div class='rec-meta'>"
                        f"<span class='genre-tag'>{row.get('track_genre', '—')}</span>"
                        f"Pop {int(row.get('popularity', 0))}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    
                    if spotify_available:
                        action_col1, action_col2 = st.columns([1, 2])
                        track_url, embed_url = _build_spotify_urls(track_id)
                        if action_col1.button(
                            "Add to Player",
                            key=f"play_scenario_{selected_scenario}_{idx}",
                            disabled=not spotify_available,
                        ):
                            _set_active_player(row["track_name"], row["artists"], embed_url)
                            st.rerun()
                        action_col2.markdown(f"[🎵 Open in Spotify]({track_url})")
                    else:
                        st.caption("❌ Spotify link unavailable.")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Custom Trajectory
# ══════════════════════════════════════════════════════════════════════════════

with tab_custom:
    st.subheader("Define Your Own Trajectory")
    st.caption("Set start and end points in the 4D feature space to generate a custom playlist.")

    col_start, col_end = st.columns(2)
    with col_start:
        st.markdown("**Start Point**")
        s_energy = st.slider("Energy", 0.0, 1.0, 0.3, 0.05, key="cs_energy")
        s_valence = st.slider("Valence", 0.0, 1.0, 0.3, 0.05, key="cs_valence")
        s_dance = st.slider("Danceability", 0.0, 1.0, 0.4, 0.05, key="cs_dance")
        s_tempo = st.slider("Tempo (BPM)", 50, 200, 90, 5, key="cs_tempo")
    with col_end:
        st.markdown("**End Point**")
        e_energy = st.slider("Energy", 0.0, 1.0, 0.8, 0.05, key="ce_energy")
        e_valence = st.slider("Valence", 0.0, 1.0, 0.7, 0.05, key="ce_valence")
        e_dance = st.slider("Danceability", 0.0, 1.0, 0.7, 0.05, key="ce_dance")
        e_tempo = st.slider("Tempo (BPM)", 50, 200, 140, 5, key="ce_tempo")

    custom_n = st.slider("Number of songs", 5, 15, 10, key="custom_n")
    custom_pop = st.slider("Popularity weight", 0.0, 1.0, 0.5, 0.1, key="custom_pop")

    start_vec = np.array([s_energy, s_valence, s_dance, (s_tempo - 50) / 150])
    end_vec = np.array([e_energy, e_valence, e_dance, (e_tempo - 50) / 150])

    if st.button("Generate Custom Playlist", type="primary", key="btn_custom"):
        st.session_state["custom_playlist"] = build_journey_playlist(
            start_vec, end_vec, traj_matrix, df_encoded,
            n_songs=custom_n, popularity_weight=custom_pop,
        )

    if "custom_playlist" in st.session_state:
        playlist = st.session_state["custom_playlist"]
        from src.journey import generate_trajectory
        waypoints = generate_trajectory(start_vec, end_vec, custom_n)

        st.subheader("Russell Emotion Space")
        fig = plot_journey(waypoints, playlist, df_encoded, start_vec, end_vec)
        st.plotly_chart(fig, width="stretch", key="custom_plot")

        st.subheader("Playlist")
        # ✨ Magical animated styles
        st.markdown(
            """
            <style>
            @keyframes aiGlow {
                0% { box-shadow: 0 0 5px rgba(200, 149, 108, 0.3); }
                50% { box-shadow: 0 0 20px rgba(200, 149, 108, 0.6), 0 0 40px rgba(74, 222, 128, 0.2); }
                100% { box-shadow: 0 0 5px rgba(200, 149, 108, 0.3); }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if len(playlist) > 0:
            first_song = playlist.iloc[0]["track_name"].replace(" ", "_").replace("/", "_")
            csv_buffer = io.StringIO()
            playlist.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            st.download_button(
                label="📥 Download Playlist as CSV",
                data=csv_data,
                file_name=f"{first_song}_recommendations.csv",
                mime="text/csv",
                key=f"download_custom",
            )
        for idx, (_, row) in enumerate(playlist.iterrows()):
            step = int(row["step"]) + 1
            track_id = _normalize_track_id(row.get("track_id"))
            spotify_available = track_id is not None
            
            # Fetch album cover if Spotify credentials are available
            album_cover_url = NO_COVER_DATA_URL
            if spotify_available:
                client_id, client_secret = _get_spotify_credentials()
                if client_id and client_secret:
                    meta = _get_spotify_track_metadata(track_id, client_id, client_secret)
                    album_cover_url = meta.get("album_cover_url", "") or NO_COVER_DATA_URL
            
            with st.container(border=True):
                card_left, card_right = st.columns([1, 4])
                with card_left:
                    st.image(album_cover_url, width=110)
                with card_right:
                    st.markdown(
                        f"<div class='rec-name'>{step}. {row['track_name']}</div>"
                        f"<div class='rec-artist'>{row['artists']}</div>"
                        f"<div class='rec-meta'>"
                        f"<span class='genre-tag'>{row.get('track_genre', '—')}</span>"
                        f"Pop {int(row.get('popularity', 0))}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    
                    if spotify_available:
                        action_col1, action_col2 = st.columns([1, 2])
                        track_url, embed_url = _build_spotify_urls(track_id)
                        if action_col1.button(
                            "Add to Player",
                            key=f"play_custom_{idx}",
                            disabled=not spotify_available,
                        ):
                            _set_active_player(row["track_name"], row["artists"], embed_url)
                            st.rerun()
                        action_col2.markdown(f"[🎵 Open in Spotify]({track_url})")
                    else:
                        st.caption("❌ Spotify link unavailable.")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Endless Radio (GMM Roaming)
# ══════════════════════════════════════════════════════════════════════════════

with tab_endless:
    st.subheader("Endless Radio — GMM-Cluster Roaming")
    st.caption(
        "Uses the GMM posterior distribution as a soft state.  Each step the "
        "algorithm samples a song whose posterior is close to the current "
        "belief, then updates the belief toward what was just played."
    )

    # Initialize state
    if "endless_history" not in st.session_state:
        st.session_state["endless_history"] = []
    if "endless_belief" not in st.session_state:
        st.session_state["endless_belief"] = None
    if "endless_excluded" not in st.session_state:
        st.session_state["endless_excluded"] = set()
    if "endless_excluded_names" not in st.session_state:
        st.session_state["endless_excluded_names"] = set()

    with st.expander("ℹ️ How the parameters shape your trajectory", expanded=False):
        st.markdown(
            "- **η (drift rate)** — single knob mixing two cost terms.\n"
            "  Score per candidate = − [η · *posterior distance* + (1−η) · *audio distance*].\n"
            "  - η = 0.05 → audio distance dominates → next song is close in "
            "(energy, valence, danceability, tempo) to the previous one (tight radio).\n"
            "  - η = 0.5 → balanced drift in cluster + audio space.\n"
            "  - η = 0.8 → posterior distance dominates → wide GMM-roam, audio can jump.\n"
            "  η also controls the EMA half-life of the seed: ≈ ln(2)/η steps.\n"
            "- **Temperature (T)** — exploit ↔ explore within the candidate pool.\n"
            "  - T = 0.1 → almost always the top-cost candidate.\n"
            "  - T = 0.5 (default) → mostly top-3, occasional 4th–10th.\n"
            "  - T = 2.0 → near-uniform across the pool.\n"
            "- **Pool size** — number of top candidates softmax samples from.\n"
            "  Smaller = tighter; larger = more variety regardless of T.\n"
            "- **Random seed** — fix it to reproduce the same trajectory."
        )

    with st.expander("⚙️ Parameters", expanded=True):
        eta = st.slider("η (belief drift rate)", 0.05, 0.8, 0.3, 0.05,
                        key="endless_eta")
        temperature = st.slider("Temperature", 0.05, 3.0, 0.5, 0.05,
                                key="endless_temp")
        pool_size = st.slider("Pool size (top_k)", 5, 50, 15, 1,
                              key="endless_pool")
        seed_input = st.text_input("Random seed (optional)",
                                   value="", key="endless_rng_seed",
                                   placeholder="leave blank for fresh randomness")

    # Seed song selection
    st.markdown("**Start from a song:**")
    seed_query = st.text_input("Search for a song or artist",
                               key="endless_search",
                               placeholder="e.g. Blinding Lights, Drake...")

    seed_idx = None
    if seed_query.strip():
        title_lower = df_encoded["track_name"].fillna("").str.lower()
        artist_lower = df_encoded["artists"].fillna("").str.lower()
        q = seed_query.strip().lower()
        matches = df_encoded[
            title_lower.str.contains(q, na=False) |
            artist_lower.str.contains(q, na=False)
        ].head(20)
        if len(matches) > 0:
            options = matches.index.tolist()
            seed_idx = st.selectbox(
                "Select song",
                options=options,
                format_func=lambda i: (
                    f"{df_encoded.iloc[i]['track_name']} — "
                    f"{df_encoded.iloc[i]['artists']} "
                    f"({df_encoded.iloc[i]['track_genre']})"
                ),
                key="endless_seed_select",
            )
        else:
            st.warning("No matches found.")

    # Display selected seed song
    if seed_idx is not None:
        seed_song = df_encoded.iloc[seed_idx]
        seed_track_id = _normalize_track_id(seed_song.get("track_id"))
        seed_spotify_available = seed_track_id is not None
        
        # Fetch album cover if Spotify credentials are available
        seed_album_cover_url = NO_COVER_DATA_URL
        if seed_spotify_available:
            client_id, client_secret = _get_spotify_credentials()
            if client_id and client_secret:
                meta = _get_spotify_track_metadata(seed_track_id, client_id, client_secret)
                seed_album_cover_url = meta.get("album_cover_url", "") or NO_COVER_DATA_URL
        
        st.markdown("**Selected Seed Song:**")
        with st.container(border=True):
            seed_card_left, seed_card_right = st.columns([1, 4])
            with seed_card_left:
                st.image(seed_album_cover_url, width=110)
            with seed_card_right:
                st.markdown(
                    f"<div class='rec-name'>{seed_song['track_name']}</div>"
                    f"<div class='rec-artist'>{seed_song['artists']}</div>"
                    f"<div class='rec-meta'>"
                    f"<span class='genre-tag'>{seed_song.get('track_genre', '—')}</span>"
                    f"Pop {int(seed_song.get('popularity', 0))}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    col_start_btn, col_reset = st.columns(2)
    with col_start_btn:
        if st.button("🎵 Start Radio", type="primary", key="btn_start_radio",
                      disabled=seed_idx is None):
            belief = gmm_result.probabilities[seed_idx].copy()
            song = df_encoded.iloc[seed_idx]
            actual = traj_matrix[seed_idx]
            st.session_state["endless_belief"] = belief
            st.session_state["endless_excluded"] = {seed_idx}
            st.session_state["endless_excluded_names"] = {song["track_name"]}
            st.session_state["endless_history"] = [{
                "track_name": song["track_name"],
                "artists": song["artists"],
                "track_genre": song["track_genre"],
                "popularity": int(song["popularity"]),
                "track_id": _normalize_track_id(song.get("track_id")),
                "actual_energy": actual[0],
                "actual_valence": actual[1],
                "actual_danceability": actual[2],
                "actual_tempo_norm": actual[3],
            }]
            st.rerun()

    with col_reset:
        if st.button("🔄 Reset Radio", key="btn_reset_radio"):
            for k in ("endless_history", "endless_belief",
                      "endless_excluded", "endless_excluded_names"):
                st.session_state.pop(k, None)
            st.rerun()

    # Active radio
    if st.session_state.get("endless_belief") is not None:
        history = st.session_state["endless_history"]
        history_df = pd.DataFrame(history)

        st.subheader("Russell Emotion Space — Drift Path")
        fig = plot_endless_history(history_df, df_encoded)
        st.plotly_chart(fig, width="stretch", key="endless_plot")

        # Per-step drift readout (belief bar chart removed per Batch 8)
        if len(history) >= 2:
            a = history[-1]; b = history[-2]
            de = a["actual_energy"] - b["actual_energy"]
            dv = a["actual_valence"] - b["actual_valence"]
            d_total = float(np.hypot(de, dv))
            col_drift, col_count = st.columns(2)
            col_drift.metric("Russell drift (last step)", f"{d_total:.3f}",
                             delta=f"ΔE={de:+.2f}, ΔV={dv:+.2f}",
                             delta_color="off")
            col_count.metric("Songs played", str(len(history)))
        else:
            st.caption(f"Songs played: {len(history)} — press Next Song to see drift.")

        if st.button("⏭️ Next Song", type="primary", key="btn_next_song"):
            with st.spinner("🔮 Predicting next drift in feature space... 🎵"):
                seed_str = seed_input.strip()
                rng = (np.random.default_rng(int(seed_str) + len(history))
                       if seed_str.isdigit() else np.random.default_rng())
                last = history[-1]
                prev_traj = np.array([last["actual_energy"], last["actual_valence"],
                                      last["actual_danceability"], last["actual_tempo_norm"]])
                idx, new_belief = gmm_endless_next(
                    belief=st.session_state["endless_belief"],
                    gmm_probs=gmm_result.probabilities,
                    traj_matrix=traj_matrix,
                    prev_traj_features=prev_traj,
                    df=df_encoded,
                    excluded=st.session_state["endless_excluded"],
                    excluded_names=st.session_state["endless_excluded_names"],
                    eta=eta,
                    temperature=temperature,
                    top_k=pool_size,
                    rng=rng,
                )
                song = df_encoded.iloc[idx]
                actual = traj_matrix[idx]
                st.session_state["endless_belief"] = new_belief
                st.session_state["endless_excluded"].add(idx)
                st.session_state["endless_excluded_names"].add(song["track_name"])
                st.session_state["endless_history"].append({
                    "track_name": song["track_name"],
                    "artists": song["artists"],
                    "track_genre": song["track_genre"],
                    "popularity": int(song["popularity"]),
                    "track_id": _normalize_track_id(song.get("track_id")),
                    "actual_energy": actual[0],
                    "actual_valence": actual[1],
                    "actual_danceability": actual[2],
                    "actual_tempo_norm": actual[3],
                })
            st.balloons()
            st.rerun()

        st.subheader("Play History")
        if len(history) > 0:
            first_song = history[0]["track_name"].replace(" ", "_").replace("/", "_")
            csv_buffer = io.StringIO()
            history_df = pd.DataFrame(history)
            history_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            st.download_button(
                label="📥 Download Play History as CSV",
                data=csv_data,
                file_name=f"{first_song}_recommendations.csv",
                mime="text/csv",
                key=f"download_endless",
            )
        for i, song in enumerate(reversed(history)):
            num = len(history) - i
            track_id = _normalize_track_id(song.get("track_id"))
            spotify_available = track_id is not None
            
            # Fetch album cover if Spotify credentials are available
            album_cover_url = NO_COVER_DATA_URL
            if spotify_available:
                client_id, client_secret = _get_spotify_credentials()
                if client_id and client_secret:
                    meta = _get_spotify_track_metadata(track_id, client_id, client_secret)
                    album_cover_url = meta.get("album_cover_url", "") or NO_COVER_DATA_URL
            
            with st.container(border=True):
                card_left, card_right = st.columns([1, 4])
                with card_left:
                    st.image(album_cover_url, width=110)
                with card_right:
                    st.markdown(
                        f"<div class='rec-name'>{num}. {song['track_name']}</div>"
                        f"<div class='rec-artist'>{song['artists']}</div>"
                        f"<div class='rec-meta'>"
                        f"<span class='genre-tag'>{song.get('track_genre', '—')}</span>"
                        f"Pop {song['popularity']}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    
                    if spotify_available:
                        action_col1, action_col2 = st.columns([1, 2])
                        track_url, embed_url = _build_spotify_urls(track_id)
                        if action_col1.button(
                            "Add to Player",
                            key=f"play_endless_{i}",
                            disabled=not spotify_available,
                        ):
                            _set_active_player(song["track_name"], song["artists"], embed_url)
                            st.rerun()
                        action_col2.markdown(f"[🎵 Open in Spotify]({track_url})")
                    else:
                        st.caption("❌ Spotify link unavailable.")
    else:
        st.info("👆 Search for a song and press **Start Radio** to begin.")
