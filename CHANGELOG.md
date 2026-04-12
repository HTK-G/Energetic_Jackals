# CHANGELOG

Track of all changes made to the Energetic Jackals Spotify ML project.

---

## 2026-04-10 — Phase 2 (K-Means & GMM) Implementation

### Phase 2A — Clustering Algorithms
- **`src/clustering.py`** — K-Means (`sklearn.cluster.KMeans`) and GMM (`sklearn.mixture.GaussianMixture`) with hyperparameter tuning. `tune_kmeans` / `tune_gmm` sweep K in [5, 30], collecting inertia/silhouette/BIC. Best K selected by highest silhouette (K-Means) or lowest BIC (GMM).

### Phase 2B — Cluster Evaluation
- **`src/evaluate.py`** — Added internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and external metrics (ARI, NMI vs `track_genre`). `metrics_comparison_table()` for side-by-side algorithm comparison.

### Phase 2C — Cluster Visualization
- **`app/page_clusters.py`** — Full cluster explorer page with 4 tabs:
  - **Cluster Visualization**: PCA and UMAP 2D scatter plots colored by cluster, with hover details. Cluster size distribution bar chart.
  - **Hyperparameter Tuning**: Elbow/silhouette/BIC charts for K selection.
  - **Cluster Profiling**: Auto-generated labels (e.g., "High-energy dance", "Acoustic"), mean feature heatmap, top genres per cluster, drill-down song list.
  - **Evaluation Metrics**: Side-by-side K-Means vs GMM comparison table with metric definitions.

### Phase 2E — Cluster-Aware Recommendation
- **`src/recommend.py`** — Added `recommend_by_cluster()` (restrict to same K-Means cluster) and `recommend_by_gmm()` (cosine similarity of GMM posterior vectors).
- **`app/page_recommend.py`** — Added recommendation mode toggle: "Embedding (KNN)", "K-Means cluster", "GMM posterior" with adjustable K.

### Dependencies Changed
- Added: `umap-learn>=0.5`

---

## 2026-04-10 — Project Restructure & Phase 1 Implementation

### Restructure
- Deleted old nested `src/` layout (`src/data/`, `src/models/`, `src/evaluation/`, `src/recommender/`, `src/utils/`, `src/visualization/`)
- Adopted flat `src/` layout per PLAN.md (`src/features.py`, `src/recommend.py`, `src/evaluate.py`, `src/explain.py`)
- Deleted old `app/streamlit_app.py`, replaced with multi-page setup (`app/app.py` + `app/page_recommend.py`)
- Removed `torch` dependency (not needed until autoencoder work)
- Added `rapidfuzz` dependency for fuzzy song search

### Phase 1 — Baseline Song-to-Song Recommendation
- **`src/features.py`** — Feature engineering: 11 audio features, sine/cosine cyclical encoding for `key` (12D vector), `StandardScaler`. Loads from `data/processed/clean_dataset_final.csv`.
- **`src/recommend.py`** — `RecommendationEngine` using `sklearn.neighbors.NearestNeighbors` with cosine metric. Fuzzy search via `rapidfuzz`. Excludes same-name duplicate songs from results.
- **`src/evaluate.py`** — Genre-based hit rate @K evaluation using `all_genres` column. Average hit rate across random sample.
- **`src/explain.py`** — Per-feature difference analysis, natural-language explanations, radar chart comparisons (query=blue, recommended=orange). Human-readable feature labels (e.g., `key_sin` → "Key (Pitch Angle)").
- **`app/app.py`** — Streamlit multi-page entry point.
- **`app/page_recommend.py`** — Song search → select from results or full catalog → top-K recommendations with similarity scores, feature comparison expanders, and radar charts.

### Repo Setup
- Added `PLAN.md` — full implementation plan (Phases 1–4)
- Added `CLAUDE.md` — project guidance for Claude Code, including fork workflow
- Added `data/processed/clean_dataset_final.csv` (89,578 rows) — merged via PR #2
- Fixed `.gitignore` to allow tracking the cleaned dataset
- Set up fork workflow: `origin` = HTK-G (group), `fork` = Jiayi459 (personal)

### Dependencies Changed
- Added: `rapidfuzz>=3.9`
- Removed: `torch>=2.6`
