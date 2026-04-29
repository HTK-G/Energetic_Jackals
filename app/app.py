"""Streamlit main entry point for the Spotify ML project."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Spotify Music Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: coffee-toned dark theme polish ───────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Headings */
h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.02em;
}

/* Accent underline for page titles */
h1 {
    border-bottom: 3px solid #c8956c;
    padding-bottom: 0.3rem;
    margin-bottom: 1rem;
}

/* Styled containers (cards) */
[data-testid="stExpander"],
div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] > div[data-testid="stContainer"] {
    border: 1px solid rgba(200, 149, 108, 0.2);
    border-radius: 12px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] > div[data-testid="stContainer"]:hover {
    border-color: rgba(200, 149, 108, 0.5);
    box-shadow: 0 0 12px rgba(200, 149, 108, 0.08);
}

/* Primary button */
button[kind="primary"] {
    background-color: #c8956c !important;
    color: #1a120d !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: background-color 0.2s ease, transform 0.1s ease;
}
button[kind="primary"]:hover {
    background-color: #d4a87e !important;
    transform: translateY(-1px);
}

/* Secondary / regular buttons */
button:not([kind="primary"]) {
    border-radius: 8px !important;
    border: 1px solid rgba(200, 149, 108, 0.3) !important;
    transition: border-color 0.2s ease;
}
button:not([kind="primary"]):hover {
    border-color: #c8956c !important;
}

/* Sidebar polish */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(200, 149, 108, 0.15);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    border-bottom: none;
    font-size: 1.1rem;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 500 !important;
    transition: color 0.2s ease;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(200, 149, 108, 0.06);
    border-radius: 10px;
    padding: 0.8rem;
}

/* Selectbox / input styling */
div[data-baseweb="select"] > div,
input[type="text"] {
    border-radius: 8px !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: #1a120d;
}
::-webkit-scrollbar-thumb {
    background: #c8956c40;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #c8956c80;
}
</style>
""", unsafe_allow_html=True)

recommend_page = st.Page("page_recommend.py", title="Song Search & Recommend", icon="\U0001F3B5", default=True)
journey_page = st.Page("page_journey.py", title="Music Journey", icon="\U0001F3BC")
model_analysis_page = st.Page("page_model_analysis.py", title="Model Analysis", icon="\U0001F4CA")

pg = st.navigation([recommend_page, journey_page, model_analysis_page])
pg.run()
