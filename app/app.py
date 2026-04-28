"""Streamlit main entry point for the Spotify ML project."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
	page_title="Spotify Music Recommender",
	layout="wide",
	initial_sidebar_state="expanded",
)

recommend_page = st.Page("page_recommend.py", title="Song Search & Recommend", icon="\U0001F3B5", default=True)
journey_page = st.Page("page_journey.py", title="Music Journey", icon="\U0001F3BC")
clusters_page = st.Page("page_clusters.py", title="Cluster Explorer", icon="\U0001F4CA")
evaluation_page = st.Page("page_evaluation.py", title="Recommendation Evaluation", icon="\U0001F4C8")

pg = st.navigation([recommend_page, journey_page, clusters_page, evaluation_page])
pg.run()
