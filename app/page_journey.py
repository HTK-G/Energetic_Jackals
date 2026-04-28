"""Music Journey page — placeholder.

Full implementation lands in Batch 4 (see PLAN_v2.md). Will replace the
Playlist-Based Recommendation page with: scenario presets, custom mode (slider
or playlist aggregation), radio mode, and Russell emotion-space visualization.
"""

from __future__ import annotations

import streamlit as st

st.title("Music Journey")
st.caption("Trajectory playlist + radio mode in Russell emotion space")

st.info(
    "**Coming soon (Batch 4).**\n\n"
    "This page will offer:\n"
    "- 6 scenario presets (workout, focus, wind down, party, commute, rainy night)\n"
    "- Custom mode: define start/end via sliders or by aggregating a song selection\n"
    "- Radio mode: continuous recommendations within a fixed feature region\n"
    "- Russell emotion-space (energy × valence) visualization with trajectory overlay"
)
