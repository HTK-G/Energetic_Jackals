# PLAN.md — Spotify Music Recommendation System

**Team**: Energetic Jackals  
**Repo**: `HTK-G/Energetic_Jackals`  
**Deliverable**: Streamlit Web App  
**Timeline**: 3–4 weeks  

---

## Dataset

- **Source**: [maharshipandya/Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Cleaned file**: `data/processed/clean_dataset_final.csv`
- **Rows**: 89,579 (deduplicated by `track_id`, multi-genre tracks aggregated)
- **Columns** (21):

| Column | Type | Role |
|---|---|---|
| `track_id` | str | Unique identifier |
| `artists` | str | Artist name(s) |
| `album_name` | str | Album name |
| `track_name` | str | Song title |
| `track_genre` | str | Primary genre label (from original dataset) |
| `popularity` | int | Spotify popularity score 0–100 |
| `explicit` | bool | Explicit content flag |
| `danceability` | float | 0.0–1.0 |
| `energy` | float | 0.0–1.0 |
| `key` | int | Musical key (0–11, pitch class) |
| `loudness` | float | dB, typically -60 to 0 |
| `mode` | int | 0 = minor, 1 = major |
| `speechiness` | float | 0.0–1.0 |
| `acousticness` | float | 0.0–1.0 |
| `instrumentalness` | float | 0.0–1.0 |
| `liveness` | float | 0.0–1.0 |
| `valence` | float | 0.0–1.0 (musical positiveness) |
| `tempo` | float | BPM |
| `time_signature` | int | Beats per measure (3, 4, 5, etc.) |
| `all_genres` | str | Comma-separated list of all genres this track belongs to |
| `num_genres` | int | Count of genres in `all_genres` |

---

## Feature Space Design

### Features used for similarity / clustering

The following **11 continuous audio features** form the core feature vector:

```
danceability, energy, loudness, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, key, mode
```

**Rationale for inclusion/exclusion**:
- `popularity` — **excluded** from feature vector. Popularity reflects external engagement, not audio similarity. Including it would bias recommendations toward already-popular tracks. However, it can be used as a downstream filter or evaluation signal.
- `time_signature` — **excluded**. Very low variance (~90%+ of tracks are 4/4). Adds noise, not signal.
- `explicit` — **excluded** from features. Binary non-audio attribute. Can be used as an optional user-side filter.
- `key` and `mode` — **included but require careful treatment**. `key` is categorical/cyclical (0–11 pitch class, wrapping around). Two encoding options:
  - Option A: Sine/cosine encoding of key (preserves cyclical distance: key 0 and key 11 are neighbors).
  - Option B: Keep as integer, acknowledge the limitation.
  - **Decision**: Use sine/cosine encoding for key. Keep mode as binary 0/1.
- `duration_ms` — **not present** in the cleaned dataset column list. Not used.

### Standardization

- **StandardScaler** (zero mean, unit variance) for all features before distance computation and clustering.
- **Justification**: Features are on different scales (e.g., `loudness` is -60 to 0 dB, `valence` is 0–1, `tempo` is 50–250 BPM). StandardScaler is preferred over MinMaxScaler because MinMaxScaler is sensitive to outliers (`loudness` and `tempo` have long tails).

### Dimensionality Reduction (for visualization only)

- **PCA** (2D/3D) — linear projection, preserves global variance structure. Good for cluster overview.
- **UMAP** (2D) — non-linear, preserves local neighborhood structure. Better for revealing fine-grained clusters.
- **Important**: Dimensionality reduction is used **only for visualization and optional clustering experiments**, NOT for the recommendation distance computation. Distances for recommendation are computed in the full standardized feature space.

---

## Implementation Phases

### Phase 1 — Baseline Song-to-Song Recommendation (Week 1)

**Goal**: User inputs a song name → system returns top-K most similar songs with similarity scores.

#### Steps

1. **Feature matrix construction**
   - Select the 11 features listed above from the cleaned dataset.
   - Apply sine/cosine encoding to `key` (produces 2 columns: `key_sin`, `key_cos`), resulting in a 12-dimensional feature vector.
   - Apply `StandardScaler`.

2. **Similarity computation**
   - Use **cosine similarity** as the primary distance metric.
   - Why cosine: it measures directional similarity in feature space, which is more meaningful than Euclidean distance when features have been standardized. Cosine similarity is also less sensitive to the magnitude of feature vectors.
   - Precompute a KNN index (e.g., `sklearn.neighbors.NearestNeighbors` with `metric='cosine'`) over the full dataset for fast retrieval.

3. **Song search & selection**
   - User types a song name → fuzzy string matching against `track_name` column (use `rapidfuzz` or `difflib`).
   - If multiple matches, display a selection list (showing artist, album, genre to disambiguate).
   - User selects one song as the query.

4. **Recommendation output**
   - Return top-K nearest neighbors (default K=10).
   - Display: song name, artist, genre, cosine similarity score.
   - **Feature-level explanation**: For each recommended song, show a comparison table or radar chart of the query song vs. the recommended song across all features. Highlight the features that are most similar and most different.

5. **Genre-based hit rate evaluation**
   - For each query song, compute the fraction of top-K recommendations that share at least one genre (from `all_genres`) with the query song.
   - Report average hit rate across a random sample of queries.
   - This is a sanity check, not a gold-standard metric — genre agreement is expected to be moderate, not perfect (since the whole motivation is that genre labels are unreliable).

#### Deliverable
- Streamlit page: search bar → song selection → recommendation list with similarity scores and feature comparison.

---

### Phase 2 — Clustering Analysis & Visualization (Week 2)

**Goal**: Cluster the full dataset using multiple algorithms. Evaluate, visualize, and profile clusters. Integrate clusters into the recommendation system.

#### 2A. Clustering Algorithms

Run all four algorithms on the same standardized 12D feature matrix:

| Algorithm | Key Hyperparameter | What It Gives |
|---|---|---|
| **K-Means** | K (number of clusters) | Hard cluster assignment. Fast, interpretable. |
| **GMM** (Gaussian Mixture Model) | K + covariance type | Soft probability vector per song (e.g., 60% cluster A, 30% cluster B). More nuanced. |
| **DBSCAN** | `eps`, `min_samples` | Density-based clusters + noise label. Does not require pre-specifying K. |
| **Agglomerative** (Hierarchical) | K or distance threshold, linkage type | Dendrogram showing hierarchical cluster relationships. |

**Hyperparameter selection**:
- K-Means / GMM / Agglomerative: Determine optimal K via:
  - Elbow method (inertia vs. K for K-Means)
  - Silhouette score vs. K
  - BIC (Bayesian Information Criterion) for GMM specifically
  - Test K in range [5, 30], given 114 genres exist but many overlap heavily.
- DBSCAN: Tune `eps` via k-distance graph (compute k-nearest-neighbor distance for each point, plot sorted, look for the elbow). `min_samples` = 2 × dimensionality is a common heuristic (so ~24).

#### 2B. Cluster Evaluation

**Internal metrics** (no ground truth needed):
- Silhouette Score — measures intra-cluster cohesion vs. inter-cluster separation. Range [-1, 1], higher is better.
- Davies-Bouldin Index — ratio of within-cluster scatter to between-cluster separation. Lower is better.
- Calinski-Harabasz Index — ratio of between-cluster variance to within-cluster variance. Higher is better.

**External metrics** (using `track_genre` as ground truth):
- Adjusted Rand Index (ARI) — measures agreement between cluster assignments and genre labels, adjusted for chance. Range [-1, 1].
- Normalized Mutual Information (NMI) — measures shared information between cluster assignments and genre labels. Range [0, 1].
- **Caveat to state in writeup**: Genre labels are imperfect ground truth. A song labeled "pop" and one labeled "dance pop" may be nearly identical in feature space. Low ARI/NMI does not necessarily mean bad clusters — it may mean the clustering captured acoustic structure that genre labels don't.

**Comparison table**: Report all metrics for all four algorithms side by side.

#### 2C. Cluster Visualization

- PCA (2D) scatter plot colored by cluster assignment. Overlay genre labels as shapes or a second coloring scheme for comparison.
- UMAP (2D) scatter plot — same treatment.
- For Agglomerative: render a dendrogram (truncated to top ~30 merges) showing cluster hierarchy.
- Interactive Streamlit widget: user can toggle between clustering algorithms and see how clusters change.

#### 2D. Cluster Profiling (Explainability)

For each cluster, compute and display:
- **Mean feature values** (radar chart per cluster, or heatmap of clusters × features).
- **Top genres** within each cluster (frequency count of `track_genre`).
- **Natural language label**: Based on the dominant features, assign a human-readable label (e.g., "High-energy dance tracks", "Acoustic/folk ballads", "Spoken word / podcast-like"). This can be done manually or via simple rule-based logic.

#### 2E. Cluster-Aware Recommendation

Extend the baseline:
- **K-Means mode**: Recommend only within the same cluster as the query song. This narrows the search space and may improve relevance.
- **GMM mode**: Use soft probability vectors. Compute similarity between songs as the cosine similarity of their GMM posterior vectors. This captures "songs that straddle similar genre boundaries" — a more nuanced signal.
- **Comparison**: Evaluate whether cluster-aware recommendation improves genre hit rate @K compared to raw KNN baseline.

#### Deliverable
- Streamlit page: interactive cluster visualization (PCA / UMAP plots), cluster profiling dashboard, toggle between algorithms.
- Recommendation page updated with cluster-aware mode option.

---

### Phase 3 — Extended Features (Week 3)

#### 3A. Playlist Input → Recommendation

- User inputs multiple songs (or a playlist).
- System computes the **aggregate feature profile** of the playlist: mean (and optionally standard deviation) of the feature vectors across all songs in the playlist.
- Recommend songs nearest to the aggregate profile that are NOT already in the playlist.
- Display: how the recommended songs compare to the playlist's average profile.

#### 3B. Mood / Scenario → Recommendation

**Approach: Manual mapping (baseline)**
- Define a set of preset moods/scenarios, each mapped to a target feature vector:

| Mood/Scenario | Target Feature Profile |
|---|---|
| Chill / Relaxing | High acousticness, low energy, medium valence, slow tempo |
| Workout / High Energy | High energy, high danceability, high tempo, low acousticness |
| Focus / Study | High instrumentalness, low speechiness, low liveness, medium tempo |
| Party | High danceability, high energy, high valence |
| Melancholic / Sad | Low valence, low energy, high acousticness |
| Road Trip | High energy, medium-high valence, high tempo |

- User selects a mood → system retrieves songs nearest to the target profile.

**Approach: Free-text scenario description (extension)**
- User types a natural language description, e.g., "I am driving in Colorado's highway on a winter morning, feeling calm."
- An LLM (or rule-based NLP) interprets the description into a target feature vector.
- System retrieves songs nearest to that vector.
- **Note**: If using an LLM, this introduces an API dependency. Could use a local lightweight model or Anthropic API. Mark as extension — only implement if time permits.

#### 3C. Feature Importance per Recommendation

For each recommended song relative to the query song:
- Compute the absolute difference per feature.
- Rank features by similarity (smallest difference = most similar).
- Display: "Recommended because: similar danceability (0.82 vs. 0.79), similar tempo (120 vs. 118 BPM). Differs in: acousticness (0.12 vs. 0.65)."
- Visualization: side-by-side radar chart of query vs. recommended song.

#### Deliverable
- Streamlit page: playlist input mode, mood selector, free-text scenario box (if extension reached).
- Feature explanation integrated into recommendation results.

---

### Phase 4 — Polish & Optional Extensions (Week 4)

#### 4A. Streamlit App Integration

- Unify all features into a cohesive multi-page Streamlit app:
  - **Page 1: Song Search & Recommend** — baseline + cluster-aware recommendation.
  - **Page 2: Playlist Analyzer** — input playlist, view aggregate profile, get recommendations.
  - **Page 3: Mood / Scenario** — preset moods + optional free-text.
  - **Page 4: Cluster Explorer** — interactive visualization, profiling, algorithm comparison.

#### 4B. Optional Extensions (implement if time allows)

| Extension | Description | Difficulty |
|---|---|---|
| **Spotify API integration** | Use Spotipy to link recommendations to Spotify URIs, auto-generate playlists in user's Spotify account. Requires Spotify developer credentials. | Medium |
| **Diversity control** | Add a diversity slider: when set high, penalize recommendations that are too similar to each other (e.g., maximal marginal relevance). Prevents returning 10 nearly identical songs. | Low–Medium |
| **Genre filter** | Allow user to constrain recommendations to specific genres or exclude genres. | Low |
| **Trainable mood mapping** | Instead of manual mood→feature mappings, learn mappings from data. E.g., collect user feedback ("this song feels chill") and train a simple regression model. Or use the genre labels + heuristic mood→genre mappings to bootstrap. | Medium–High |
| **Multi-seed input with feature trajectory** | Input an ordered list of songs and generate a playlist that smoothly transitions between them in feature space (interpolation). | Medium |

---

## Evaluation Summary

| What | Metric | Used Where |
|---|---|---|
| Cluster quality (internal) | Silhouette, Davies-Bouldin, Calinski-Harabasz | Phase 2 — compare clustering algorithms |
| Cluster quality (external) | ARI, NMI vs. `track_genre` | Phase 2 — validate against genre labels |
| Recommendation relevance | Genre hit rate @K | Phase 1 & 2 — sanity check |
| Recommendation quality | Feature-level comparison | Phase 3C — explainability |
| Algorithm comparison | All above metrics, side by side for K-Means, GMM, DBSCAN, Agglomerative | Phase 2 |

---

## Technical Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| Data processing | pandas, numpy |
| ML / Clustering | scikit-learn (KMeans, GaussianMixture, DBSCAN, AgglomerativeClustering, NearestNeighbors) |
| Dimensionality reduction | scikit-learn (PCA), umap-learn (UMAP) |
| Visualization | matplotlib, seaborn, plotly (for interactive Streamlit charts) |
| String matching | rapidfuzz |
| Web app | Streamlit |
| Spotify API (optional) | spotipy |
| Version control | Git / GitHub |

---

## Project Structure (Proposed)

```
Energetic_Jackals/
├── data/
│   ├── raw/                          # Original dataset
│   └── processed/
│       └── clean_dataset_final.csv   # Cleaned 89,579 rows
├── notebooks/                        # Exploratory analysis, prototyping
│   ├── 01_eda.ipynb
│   ├── 02_clustering.ipynb
│   └── 03_recommendation.ipynb
├── src/
│   ├── features.py                   # Feature engineering, scaling, encoding
│   ├── clustering.py                 # K-Means, GMM, DBSCAN, Agglomerative
│   ├── recommend.py                  # KNN-based + cluster-aware recommendation
│   ├── evaluate.py                   # Internal/external metrics, hit rate
│   ├── explain.py                    # Feature-level explanation, radar charts
│   └── mood.py                       # Mood/scenario → feature vector mapping
├── app/
│   ├── app.py                        # Streamlit main entry
│   ├── page_recommend.py             # Song search & recommend page
│   ├── page_playlist.py              # Playlist analyzer page
│   ├── page_mood.py                  # Mood / scenario page
│   └── page_clusters.py             # Cluster explorer page
├── PLAN.md                           # This file
├── README.md
└── requirements.txt
```

---

## Weekly Milestones

| Week | Deliverable | Key Risk |
|---|---|---|
| 1 | Phase 1 complete: baseline KNN recommendation with Streamlit search UI, similarity scores, feature comparison display | Fuzzy matching performance on 89K track names; deciding cosine vs. euclidean |
| 2 | Phase 2 complete: all 4 clustering algorithms trained, evaluated, visualized. Cluster-aware recommendation integrated | DBSCAN may label most points as noise in 12D; optimal K determination |
| 3 | Phase 3 complete: playlist input, mood presets, feature explainability | Mood→feature mapping quality; free-text parsing complexity |
| 4 | Phase 4: Streamlit polish, optional extensions, final report/presentation | Time pressure on optional features |
