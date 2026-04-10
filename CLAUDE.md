# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spotify ML project: loads Spotify audio features (~130k songs), preprocesses them with sklearn pipelines, learns embeddings (PCA or PyTorch autoencoder), clusters with a from-scratch NumPy K-Means, and recommends songs via cosine similarity. Streamlit UI ties it all together.

## Commands

```bash
# Setup
uv sync

# Run the Streamlit app
uv run streamlit run app/streamlit_app.py
```

No test framework is configured yet. No linter/formatter config exists.

## Architecture

The pipeline flows: **loader** -> **preprocessing** -> **embeddings** -> **clustering** -> **recommendation**.

- `src/data/loader.py` — Loads CSV from `data/raw/` or generates a deterministic demo dataset via `src/utils/demo_data.py`. Normalizes column names through `SCHEMA_ALIASES` to handle variant CSV schemas.
- `src/data/preprocessing.py` — Builds an sklearn `ColumnTransformer` (median impute + StandardScaler for continuous, mode impute + OneHotEncoder for categorical). Returns a `PreparedData` dataclass with the cleaned DataFrame and feature matrix.
- `src/models/embeddings.py` — `compute_song_embeddings()` dispatches to PCA or autoencoder. Returns `EmbeddingResult` containing embeddings, model, and a 2D PCA projection for visualization.
- `src/models/autoencoder.py` — Feed-forward autoencoder in PyTorch (encoder/decoder with configurable hidden dims). Trains on CPU only.
- `src/models/kmeans.py` — `NumpyKMeans`: pure NumPy K-Means with K-Means++ init and empty-cluster recovery. Intentionally avoids sklearn's KMeans.
- `src/recommender/similarity.py` — Cosine similarity utilities for nearest-neighbor retrieval.
- `src/recommender/recommend.py` — `RecommendationEngine`: wraps catalog + embeddings + cluster labels. Supports embedding-based and cluster-constrained recommendation modes.
- `app/streamlit_app.py` — `build_pipeline()` orchestrates the full flow. UI has four tabs: Song Explorer, Recommendation, Clustering Visualization, Embedding Explorer.

## Key Design Decisions

- K-Means is implemented from scratch in NumPy — do not replace with sklearn's KMeans.
- Feature columns are defined in `src/utils/constants.py` (`CONTINUOUS_AUDIO_FEATURES`, `CATEGORICAL_AUDIO_FEATURES`). Add new features there.
- The app works without any dataset by falling back to a generated demo catalog (seed=42).
- All imports use absolute paths from `src.*` (e.g., `from src.data.loader import ...`).
- Python >=3.11 required. Uses `uv` for dependency management.

## Fork Workflow

This repo uses a fork workflow. Remotes:
- **`origin`** — `HTK-G/Energetic_Jackals` (shared group repo)
- **`fork`** — `Jiayi459/Energetic_Jackals` (personal fork)

```bash
# Sync with latest group code
git checkout master
git pull origin master

# Create feature branch, work, commit
git checkout -b my-feature

# Push to YOUR fork (not origin)
git push -u fork my-feature

# Open PR: Jiayi459:my-feature → HTK-G:master
gh pr create --repo HTK-G/Energetic_Jackals
```

Do not push directly to `origin`. Always push feature branches to `fork` and merge via PR.
