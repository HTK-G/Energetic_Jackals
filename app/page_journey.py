"""Music Journey — trajectory playlists and GMM-cluster endless radio.

Tab 1: Scenario Presets — 6 data-driven scenarios with trajectory playlists
Tab 2: Custom Mode — user-defined start/end via sliders → trajectory playlist
Tab 3: Endless Radio — GMM-cluster roaming with Bayesian belief update
"""

from __future__ import annotations

from pathlib import Path

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

        # Song list
        st.subheader("Playlist")
        for _, row in playlist.iterrows():
            step = int(row["step"]) + 1
            with st.container(border=True):
                c1, c2 = st.columns([0.5, 4])
                with c1:
                    st.markdown(f"### {step}")
                with c2:
                    st.markdown(f"**{row['track_name']}**")
                    st.caption(
                        f"{row['artists']} · {row['track_genre']} · "
                        f"Pop {row['popularity']}"
                    )

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
        for _, row in playlist.iterrows():
            step = int(row["step"]) + 1
            with st.container(border=True):
                c1, c2 = st.columns([0.5, 4])
                with c1:
                    st.markdown(f"### {step}")
                with c2:
                    st.markdown(f"**{row['track_name']}**")
                    st.caption(
                        f"{row['artists']} · {row['track_genre']} · "
                        f"Pop {row['popularity']}"
                    )

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
                "actual_energy": actual[0],
                "actual_valence": actual[1],
                "actual_danceability": actual[2],
                "actual_tempo_norm": actual[3],
            })
            st.rerun()

        st.subheader("Play History")
        for i, song in enumerate(reversed(history)):
            num = len(history) - i
            with st.container(border=True):
                c1, c2 = st.columns([0.5, 4])
                with c1:
                    st.markdown(f"### {num}")
                with c2:
                    st.markdown(f"**{song['track_name']}**")
                    st.caption(
                        f"{song['artists']} · {song['track_genre']} · "
                        f"Pop {song['popularity']}"
                    )
    else:
        st.info("👆 Search for a song and press **Start Radio** to begin.")
