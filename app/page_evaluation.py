"""Recommendation Evaluation page — placeholder.

Full implementation lands in Batch 5 (see PLAN_v2.md). Will display the
6-method × 2-metric comparison: KNN / K-Means cluster / GMM posterior vs
random / popularity / genre_match baselines, on Genre Hit Rate and the
new Genre Coverage metric.
"""

from __future__ import annotations

import streamlit as st

st.title("Recommendation Evaluation")
st.caption("6 methods × 2 metrics comparison")

st.info(
    "**Coming soon (Batch 5).**\n\n"
    "This page will compare three of our recommendation algorithms (KNN, "
    "K-Means cluster, GMM posterior) against three trivial baselines (random, "
    "popularity, genre-match) on:\n"
    "- **Genre Hit Rate** — fraction of top-K recommendations sharing a genre with the query\n"
    "- **Genre Coverage** — number of distinct genres in the top-K (a new metric)\n\n"
    "Results will be precomputed offline by `scripts/evaluate_recommendations.py`."
)
