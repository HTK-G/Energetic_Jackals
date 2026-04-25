"""Precompute heavy training artifacts for the Streamlit app.

Run once after setup:
    uv run python -m scripts.precompute

Re-run after dataset or model changes:
    uv run python -m scripts.precompute --force
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import joblib
from sklearn.decomposition import PCA

from src.clustering import fit_gmm, fit_kmeans, tune_gmm, tune_kmeans
from src.evaluate import evaluate_clustering
from src.features import build_feature_matrix, load_dataset

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
KMEANS_K_RANGE = range(5, 31)
GMM_K_RANGE = range(5, 51)


def _step(label: str, path: Path, force: bool, fn: Callable):
    """Run `fn` and persist to `path`, or load if it already exists."""
    if path.exists() and not force:
        print(f"[skip] {label}  ({path.name} exists)")
        return joblib.load(path)
    t0 = time.time()
    print(f"[ run] {label} ...")
    obj = fn()
    joblib.dump(obj, path)
    print(f"[done] {label}  ({time.time() - t0:.1f}s)  -> {path.name}")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute even if artifact exists")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts directory: {ARTIFACTS_DIR}\n")
    overall_start = time.time()

    feats = _step(
        "Feature matrix",
        ARTIFACTS_DIR / "feature_matrix.joblib",
        args.force,
        lambda: _build_features(),
    )
    df_encoded = feats["df_encoded"]
    feature_matrix = feats["feature_matrix"]

    _step(
        "PCA 2D projection",
        ARTIFACTS_DIR / "pca_2d.joblib",
        args.force,
        lambda: PCA(n_components=2, random_state=42).fit_transform(feature_matrix),
    )

    tuning_kmeans = _step(
        f"Tune K-Means (sklearn, K in [{KMEANS_K_RANGE.start}, {KMEANS_K_RANGE.stop - 1}])",
        ARTIFACTS_DIR / "tuning_kmeans.joblib",
        args.force,
        lambda: tune_kmeans(feature_matrix, k_range=KMEANS_K_RANGE),
    )

    kmeans_best = _step(
        f"Fit CustomKMeans at best K={tuning_kmeans.best_k}",
        ARTIFACTS_DIR / "kmeans_best.joblib",
        args.force,
        lambda: fit_kmeans(feature_matrix, n_clusters=tuning_kmeans.best_k),
    )

    tuning_gmm_full = _step(
        f"Tune GMM (full covariance, K in [{GMM_K_RANGE.start}, {GMM_K_RANGE.stop - 1}])",
        ARTIFACTS_DIR / "tuning_gmm_full.joblib",
        args.force,
        lambda: tune_gmm(feature_matrix, k_range=GMM_K_RANGE, covariance_type="full"),
    )

    tuning_gmm_diag = _step(
        f"Tune GMM (diagonal covariance, K in [{GMM_K_RANGE.start}, {GMM_K_RANGE.stop - 1}])",
        ARTIFACTS_DIR / "tuning_gmm_diag.joblib",
        args.force,
        lambda: tune_gmm(feature_matrix, k_range=GMM_K_RANGE, covariance_type="diag"),
    )

    gmm_full_best = _step(
        f"Fit GMM (full) at best K={tuning_gmm_full.best_k}",
        ARTIFACTS_DIR / "gmm_full_best.joblib",
        args.force,
        lambda: fit_gmm(
            feature_matrix, n_clusters=tuning_gmm_full.best_k, covariance_type="full"
        ),
    )

    gmm_diag_best = _step(
        f"Fit GMM (diag) at best K={tuning_gmm_diag.best_k}",
        ARTIFACTS_DIR / "gmm_diag_best.joblib",
        args.force,
        lambda: fit_gmm(
            feature_matrix, n_clusters=tuning_gmm_diag.best_k, covariance_type="diag"
        ),
    )

    _step(
        "Metrics comparison (3 algorithms)",
        ARTIFACTS_DIR / "metrics_comparison.joblib",
        args.force,
        lambda: [
            evaluate_clustering(
                "K-Means", kmeans_best.n_clusters, feature_matrix,
                kmeans_best.labels, df_encoded["track_genre"],
            ),
            evaluate_clustering(
                "GMM (full)", gmm_full_best.n_clusters, feature_matrix,
                gmm_full_best.labels, df_encoded["track_genre"],
            ),
            evaluate_clustering(
                "GMM (diag)", gmm_diag_best.n_clusters, feature_matrix,
                gmm_diag_best.labels, df_encoded["track_genre"],
            ),
        ],
    )

    print(f"\nTotal: {time.time() - overall_start:.1f}s")
    print("All artifacts ready.")


def _build_features() -> dict:
    df = load_dataset()
    feature_matrix, scaler, df_encoded = build_feature_matrix(df)
    return {
        "df_encoded": df_encoded,
        "feature_matrix": feature_matrix,
        "scaler": scaler,
    }


if __name__ == "__main__":
    main()
