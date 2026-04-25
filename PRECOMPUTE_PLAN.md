# Implementation Plan — Performance Optimization (Precompute + Pickle)

## Overall Goal

App startup: **30–80 minutes → 2–5 seconds**.

All training is moved to `scripts/precompute.py` and run **once offline**. Results
are persisted to `artifacts/`. The Streamlit app only loads pickled artifacts.

## Time Budget

| Step | Task | Est. Time |
|---|---|---|
| 1 | Project structure (`scripts/`, `artifacts/`) | 5 min |
| 2 | Write `scripts/precompute.py` | 40 min |
| 3 | Refactor `app/page_clusters.py` to load artifacts | 30 min |
| 4 | Refactor `src/clustering.py` (sklearn for tuning, CustomKMeans for final fit) | 20 min |
| 5 | Run `precompute.py` to validate (5–15 min wait) | 15 min |
| 6 | Update `README.md` and `.gitignore` | 10 min |
| **Total** | | **~2 hours** |

---

## Step 1 — Project Structure

```
Energetic_Jackals/
├── app/
│   ├── app.py
│   ├── page_recommend.py
│   └── page_clusters.py
├── src/
│   ├── __init__.py
│   ├── clustering.py
│   ├── custom_kmeans.py
│   ├── evaluate.py
│   ├── features.py
│   ├── recommend.py
│   └── explain.py
├── scripts/                        ← NEW
│   └── precompute.py               ← NEW
├── artifacts/                      ← NEW (gitignored)
│   ├── feature_matrix.joblib
│   ├── tuning_kmeans.joblib
│   ├── tuning_gmm_full.joblib
│   ├── tuning_gmm_diag.joblib
│   ├── kmeans_best.joblib
│   ├── gmm_full_best.joblib
│   ├── gmm_diag_best.joblib
│   ├── pca_2d.joblib
│   └── metrics_comparison.joblib
├── data/
├── PLAN.md
├── PRECOMPUTE_PLAN.md
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Step 2 — `scripts/precompute.py`

Design principles:
- CLI: `python -m scripts.precompute` (or `uv run python -m scripts.precompute`)
- Reentrant: skip existing artifacts unless `--force`
- Progress logging: print elapsed time per step
- Three GMMs: full and diag covariance (per Q answered)
- K range: `range(5, 31)` (full search)

Sequence:
1. Load dataset → build feature matrix → save `feature_matrix.joblib`
2. PCA 2D projection → save `pca_2d.joblib`
3. Tune K-Means via sklearn over K range → save `tuning_kmeans.joblib`
4. Fit final K-Means with **CustomKMeans** at best K → save `kmeans_best.joblib`
5. Tune GMM (full) over K range → save `tuning_gmm_full.joblib`
6. Tune GMM (diag) over K range → save `tuning_gmm_diag.joblib`
7. Fit final GMM-full at best K → save `gmm_full_best.joblib`
8. Fit final GMM-diag at best K → save `gmm_diag_best.joblib`
9. Compute all evaluation metrics → save `metrics_comparison.joblib`

---

## Step 3 — `app/page_clusters.py` Refactor

Changes:
- Drop inline `_tune_kmeans`, `_tune_gmm`, `_compute_projections`. Replace with `joblib.load(...)`.
- Drop K slider (Q8 confirmed: don't let users change K).
- Algorithm selector now has **3 options**: K-Means (CustomKMeans) / GMM (full) / GMM (diag).
- Tab 4 metrics: load precomputed list directly.
- Drop UMAP entirely.

Open question: **Q9** — what to do with the PCA/UMAP radio after removing UMAP?
- (a) Drop the radio, show PCA only — _recommended_
- (b) Replace UMAP option with another view (e.g., feature heatmap)

---

## Step 4 — `src/clustering.py` Refactor

Changes:
- `tune_kmeans` uses **sklearn KMeans** internally (faster, comparable quality for K selection).
- `fit_kmeans` continues using **CustomKMeans** (course requirement).
- `tune_gmm` and `fit_gmm` accept `covariance_type` already — confirm callsite.

Open question: **Q10** — where does tuning logic live?
- (a) Inline tuning in `precompute.py`, delete `tune_*` from `clustering.py`
- (b) Keep `tune_*` in `clustering.py`, `precompute.py` calls them — _recommended_

---

## Step 5 — Validate Precompute

Expected timings (one-time):
- Features + PCA: ~30 s
- K-Means tuning (sklearn × 26 K): 1–3 min
- CustomKMeans final fit at best K: 30 s – 2 min
- GMM full tuning (26 K, n_init=1): 5–10 min
- GMM diag tuning (26 K, n_init=1): 1–3 min
- GMM final fits: 1–2 min
- Metrics: 30 s
- **Total**: 10–20 min

---

## Step 6 — README + .gitignore

- Add `artifacts/` to `.gitignore`.
- Add a "Precompute artifacts" section to `README.md` explaining `python -m scripts.precompute` must be run once after `uv sync`.

---

## Open Questions (for confirmation)

| # | Question | Recommendation |
|---|---|---|
| Q9 | After removing UMAP, what to do with PCA/UMAP radio in Tab 1? | (a) Drop the radio, show PCA only |
| Q10 | Where does tuning logic live? | (b) Keep in `clustering.py`, `precompute.py` orchestrates |

## Additional Observations / Advice

1. **`page_recommend.py` also fits clusters on demand** (lines 168–171, K slider at 340). Plan misses this. To hit the 2–5 s startup goal we should either (i) make it also load precomputed K-Means/GMM at the fixed best K, or (ii) keep it but cache aggressively. Recommendation: (i) — load precomputed, drop K slider there too.
2. **`joblib` should be added to `pyproject.toml`** dependencies explicitly (currently transitive via sklearn).
3. **K-Means best K vs GMM best K may differ**. Plan handles this correctly by saving separate `*_best.joblib` per algorithm.
4. The `umap-learn` dependency can be **removed from `pyproject.toml`** once UMAP is gone.
