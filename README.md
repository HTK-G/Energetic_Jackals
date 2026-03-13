# Spotify ML Project

A modular machine learning project for exploring, clustering, and recommending songs from the Spotify Audio Features dataset.

Dataset source: https://www.kaggle.com/datasets/tomigelo/spotify-audio-features

The system is built around the Spotify Audio Features dataset containing roughly 130k songs with features such as danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, popularity, key, mode, and time signature.

## What The Project Does

- Loads the Kaggle Spotify dataset from `data/raw/` or falls back to a deterministic demo dataset.
- Cleans and preprocesses metadata and audio features into a reusable feature matrix.
- Learns song embeddings with either PCA or a PyTorch autoencoder.
- Clusters songs using a NumPy implementation of K-Means built from scratch.
- Recommends related tracks through cosine similarity in embedding space or cluster-constrained retrieval.
- Visualizes clusters, embeddings, feature correlations, and song-level audio profiles in Streamlit.

## Repository Layout

```text
spotify-ml-project
├── app/
│   └── streamlit_app.py
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── embeddings.py
│   │   └── kmeans.py
│   ├── recommender/
│   │   ├── recommend.py
│   │   └── similarity.py
│   ├── utils/
│   │   ├── constants.py
│   │   └── demo_data.py
│   └── visualization/
│       └── plots.py
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Dataset Format

The project expects a CSV with Spotify-style columns. The loader normalizes common variants such as `name` and `track_name`, and supports feature columns including:

- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `duration_ms`
- `popularity`
- `key`
- `mode`
- `time_signature`

Helpful metadata columns include song name, artist, album, and a track identifier. If some metadata is missing, the loader fills in safe defaults.

## Setup With uv

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv lock
```

You can also use uv's project workflow directly:

```bash
uv sync
```

## Run The App

Place the Kaggle CSV inside `data/raw/`, for example:

```text
data/raw/spotify_audio_features.csv
```

Then start the Streamlit UI:

```bash
uv run streamlit run app/streamlit_app.py
```

If no CSV is present, the app still runs end-to-end using a generated demo catalog.

## How Recommendation Works

1. Audio features are cleaned, imputed, scaled, and encoded into a feature matrix.
2. The project learns low-dimensional song embeddings using PCA or an autoencoder.
3. Recommendations are produced by cosine similarity in embedding space.
4. Cluster-based mode restricts candidates to songs from the same learned cluster before ranking by similarity.

## How Clustering Works

The clustering module implements K-Means directly in NumPy rather than calling `sklearn.cluster.KMeans`.

The implementation includes:

- centroid initialization with random or K-Means++ style seeding
- assignment step through Euclidean distance minimization
- centroid recomputation for each cluster
- empty-cluster recovery logic
- convergence checking through centroid movement tolerance

## Main Modules

- `src/data`: dataset loading, schema normalization, preprocessing, and feature construction
- `src/models`: PCA embeddings, PyTorch autoencoder embeddings, and scratch K-Means clustering
- `src/recommender`: cosine similarity utilities and recommendation engine
- `src/evaluation`: cluster and embedding quality metrics
- `src/visualization`: Plotly charts for the app and notebook workflows

## Extending The Project

The scaffold is intentionally modular so you can add:

- contrastive or self-supervised representation learning
- approximate nearest neighbor search
- hybrid recommenders that combine popularity or metadata priors
- cluster explainability and feature attribution
- offline experiment tracking and model persistence
