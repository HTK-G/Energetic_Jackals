# Spotify Music Recommendation System

**Team**: Energetic Jackals
**Repo**: `HTK-G/Energetic_Jackals`

A Streamlit web app that recommends songs based on audio feature similarity, with clustering analysis and visualization. Built on the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (~89,500 deduplicated tracks).

## Features

- **Song Search & Recommend** — Fuzzy search by song name or artist, then get top-K similar songs with cosine similarity scores and per-feature explanations (radar charts, text breakdowns).
- **Two Recommendation Modes**:
  - _K-Means cluster_ — Restrict recommendations to songs in the same cluster.
  - _GMM posterior_ — Rank songs by cosine similarity of their soft cluster membership vectors.
- **Music Journey** — Sequential playlist generation and GMM-cluster roaming:
  - _Scenario Presets_ — 6 data-driven scenarios (workout, focus, wind down, party, commute, rainy night) with start/end feature vectors derived from genre-matched song percentiles.
  - _Custom Trajectory_ — User-defined start/end via sliders; greedy sequential nearest-neighbor with composite scoring generates playlists.
  - _Endless Radio (GMM)_ — GMM-cluster roaming with Bayesian belief update: each song shifts the posterior belief vector, creating a probabilistic drift through feature space.
  - _Russell Emotion Space_ — 2D energy×valence visualization with trajectory overlays.
- **Model Analysis** — Hyperparameter tuning charts (elbow, silhouette, BIC), cluster profiling with auto-generated labels and genre breakdowns, cluster overlap heatmap (K-Means × GMM-full contingency matrix), and 5-method recommendation evaluation (K-Means cluster, GMM posterior vs. random, popularity, genre-match baselines) across genre hit rate and genre coverage metrics.

## Setup

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Precompute Artifacts (one-time)

All heavy training (K-Means tuning, GMM tuning with full + diag covariance,
final fits, PCA projection, evaluation metrics) runs **offline** and is
persisted to `artifacts/`. The Streamlit app only loads pickles, so startup
drops from tens of minutes to a few seconds.

```bash
uv run python -m scripts.precompute
uv run python -m scripts.derive_scenario_mappings
```

This takes ~10–20 minutes the first time. Re-run with `--force` if the dataset
changes:

```bash
uv run python -m scripts.precompute --force
uv run python -m scripts.derive_scenario_mappings
```

The `artifacts/` directory is gitignored — each developer regenerates it locally.

## Run the App

```bash
uv run streamlit run app/app.py
```

The app loads from `data/processed/clean_dataset_final.csv` and from
`artifacts/`. If artifacts are missing, the app prints a clear error pointing
back to `scripts.precompute`.

### Spotify API Setup (Optional)

To enable album art, metadata enrichment, and direct playback in song cards, set up Spotify Developer credentials:

1. **Create a Spotify Developer account** at https://developer.spotify.com/dashboard
2. **Create a new app** and accept the terms
3. **Copy your credentials**:
   - `Client ID`
   - `Client Secret`
4. **Create `.streamlit/secrets.toml`** in the project root:
   ```toml
   SPOTIFY_CLIENT_ID = "your_client_id_here"
   SPOTIFY_CLIENT_SECRET = "your_client_secret_here"
   ```
5. **Restart the Streamlit app** — credentials will load automatically

> **Note**: The app works without Spotify credentials (graceful fallback), but album covers and embedded playback won't display.

## Recent Updates

- **Model Analysis page** — unified clustering + recommendation evaluation page with 5-method comparison (K-Means cluster, GMM posterior vs. three baselines), genre hit rate and genre coverage metrics, and cluster overlap heatmap.
- **Music Journey page** with trajectory playlists (greedy sequential NN), GMM-cluster roaming endless radio, and Russell emotion-space visualization.
- **Dark coffee theme** — elegant dark brown UI with warm caramel accents and Inter font.
- Added Spotify-powered song cards with album art, richer metadata, and direct playback controls.

## Dataset

- **Source**: [maharshipandya/Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Cleaned file**: `data/processed/clean_dataset_final.csv` (89,578 rows, 21 columns)
- **Key columns**: `track_name`, `artists`, `album_name`, `track_genre`, `all_genres`, plus 11 audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key, mode)

## Feature Engineering

The 11 audio features are transformed into a 12D standardized vector:

- `key` (0–11 pitch class) is replaced with sine/cosine encoding (2 columns) to preserve cyclical distance.
- `mode` is kept as binary (0 = minor, 1 = major).
- All features are standardized with `StandardScaler`.
- `popularity`, `time_signature`, `explicit`, `duration_ms` are excluded from the feature vector.

## Project Structure

```
Energetic_Jackals/
├── app/
│   ├── app.py                  # Streamlit multi-page entry point + theme CSS
│   ├── page_recommend.py       # Song search & recommendation page
│   ├── page_journey.py         # Music Journey (trajectory + GMM endless radio)
│   └── page_model_analysis.py  # Clustering analysis + recommendation evaluation
├── scripts/
│   ├── precompute.py           # Offline training: tuning, final fits, PCA, metrics
│   └── derive_scenario_mappings.py  # Data-driven scenario feature vectors
├── artifacts/                  # Pickled outputs of precompute (gitignored)
├── .streamlit/
│   └── config.toml             # Dark coffee theme
├── data/
│   └── processed/
│       └── clean_dataset_final.csv
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── features.py             # Feature engineering, encoding, scaling
│   ├── recommend.py            # Cluster-aware recommendation engine
│   ├── journey.py              # Trajectory generation + GMM-cluster roaming
│   ├── visualization.py        # Russell emotion-space plots
│   ├── clustering.py           # K-Means and GMM with hyperparameter tuning
│   ├── custom_kmeans.py        # From-scratch NumPy K-Means (course requirement)
│   ├── evaluate.py             # Genre hit rate, internal cluster metrics
│   └── explain.py              # Feature comparison, radar charts, explanations
├── pyproject.toml
└── requirements.txt
```

## Implementation Status

| Phase | Description                                                                                                      | Status |
| ----- | ---------------------------------------------------------------------------------------------------------------- | ------ |
| 1     | Baseline song-to-song recommendation (fuzzy search, feature explanations)                                        | Done   |
| 2     | Clustering analysis — K-Means & GMM (full + diag covariance), evaluation, cluster-aware recommendation           | Done   |
| 3     | Music Journey — trajectory playlists, GMM-cluster roaming, Russell emotion-space visualization                    | Done   |
| 4     | Model Analysis — 5-method recommendation evaluation, cluster overlap heatmap, Silhouette comparison              | Done   |

## Technical Stack

| Component                | Tool                                                     |
| ------------------------ | -------------------------------------------------------- |
| Language                 | Python 3.11+                                             |
| Data processing          | pandas, numpy                                            |
| ML / Clustering          | scikit-learn (GaussianMixture, NearestNeighbors)         |
| Custom algorithm         | NumPy K-Means from scratch                               |
| Dimensionality reduction | scikit-learn (PCA)                                       |
| Persistence              | joblib (precomputed artifacts)                           |
| Visualization            | plotly (interactive Streamlit charts)                    |
| String matching          | rapidfuzz                                                |
| Web app                  | Streamlit                                                |
