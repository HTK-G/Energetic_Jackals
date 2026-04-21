"""Streamlit main entry point for the Spotify ML project."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
	page_title="Spotify Music Recommender",
	layout="wide",
	initial_sidebar_state="expanded",
)

recommend_page = st.Page("page_recommend.py", title="Song Search & Recommend", icon="\U0001F3B5", default=True)
clusters_page = st.Page("page_clusters.py", title="Cluster Explorer", icon="\U0001F4CA")

pg = st.navigation([recommend_page, clusters_page])
pg.run()
