# Spotify Music Recommendation System

**Team**: Energetic Jackals
**Repo**: `HTK-G/Energetic_Jackals`

A Streamlit web app that recommends songs based on audio feature similarity, with clustering analysis and visualization. Built on the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (~89,500 deduplicated tracks).

## Features

- **Song Search & Recommend** — Fuzzy search by song name or artist, then get top-K similar songs with cosine similarity scores and per-feature explanations (radar charts, text breakdowns).
- **Three Recommendation Modes**:
  - _Embedding (KNN)_ — Nearest neighbors in the full 12D standardized feature space.
  - _K-Means cluster_ — Restrict recommendations to songs in the same cluster.
  - _GMM posterior_ — Rank songs by cosine similarity of their soft cluster membership vectors.
- **Cluster Explorer** — Interactive PCA/UMAP scatter plots, hyperparameter tuning charts (elbow, silhouette, BIC), cluster profiling with auto-generated labels and genre breakdowns, evaluation metrics comparison.

## Setup

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run the App

```bash
uv run streamlit run app/app.py
```

The app loads from `data/processed/clean_dataset_final.csv`.

## Recent Updates

- Added Spotify-powered song cards with album art, richer metadata, and direct playback controls.
- Integrated search-driven seed selection so users can search and immediately choose the song they want to recommend from.
- Moved the Spotify player into the sidebar and improved the recommendation card experience with embedded feature comparisons.

## Dataset

- **Source**: [maharshipandya/Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Cleaned file**: `data/processed/clean_dataset_final.csv` (89,578 rows, 21 columns)
- **Key columns**: `track_name`, `artists`, `album_name`, `track_genre`, `all_genres`, plus 11 audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key, mode)

## Feature Engineering

The 11 audio features are transformed into a 12D standardized vector:

- `key` (0–11 pitch class) is replaced with sine/cosine encoding (2 columns) to preserve cyclical distance.
- `mode` is kept as binary (0 = minor, 1 = major).
- All features are standardized with `StandardScaler`.
- `popularity`, `time_signature`, `explicit`, `duration_ms` are excluded from the feature vector (see PLAN.md for rationale).

## Project Structure

```
Energetic_Jackals/
├── app/
│   ├── app.py                  # Streamlit multi-page entry point
│   ├── page_recommend.py       # Song search & recommendation page
│   └── page_clusters.py        # Cluster explorer page
├── data/
│   └── processed/
│       └── clean_dataset_final.csv
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── features.py             # Feature engineering, encoding, scaling
│   ├── recommend.py            # KNN + cluster-aware recommendation engine
│   ├── clustering.py           # K-Means and GMM with hyperparameter tuning
│   ├── evaluate.py             # Genre hit rate, internal/external cluster metrics
│   └── explain.py              # Feature comparison, radar charts, explanations
├── PLAN.md                     # Full implementation plan (Phases 1–4)
├── CHANGELOG.md                # Track of all changes
├── CLAUDE.md                   # Claude Code guidance
├── pyproject.toml
└── requirements.txt
```

## Implementation Status

| Phase | Description                                                                                                      | Status  |
| ----- | ---------------------------------------------------------------------------------------------------------------- | ------- |
| 1     | Baseline song-to-song recommendation (KNN, fuzzy search, feature explanations)                                   | Done    |
| 2     | Clustering analysis — K-Means & GMM (2 of 4 algorithms), evaluation, visualization, cluster-aware recommendation | Done    |
| 2     | Clustering — DBSCAN & Agglomerative                                                                              | Planned |
| 3     | Playlist input, mood/scenario recommendation, feature importance                                                 | Planned |
| 4     | App polish, optional extensions (Spotify API, diversity control, etc.)                                           | Planned |

## Technical Stack

| Component                | Tool                                                     |
| ------------------------ | -------------------------------------------------------- |
| Language                 | Python 3.11+                                             |
| Data processing          | pandas, numpy                                            |
| ML / Clustering          | scikit-learn (KMeans, GaussianMixture, NearestNeighbors) |
| Dimensionality reduction | scikit-learn (PCA), umap-learn (UMAP)                    |
| Visualization            | plotly (interactive Streamlit charts)                    |
| String matching          | rapidfuzz                                                |
| Web app                  | Streamlit                                                |
