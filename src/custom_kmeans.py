import numpy as np


class CustomKMeans:
    def __init__(
        self,
        n_clusters: int,
        max_iters: int = 100,
        tol: float = 1e-4,
        random_state=None,
        n_init: int = 10,
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init

        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.random_state = random_state

    # ── Initialization ────────────────────────────────────────────────────────

    def _kmeans_plus_plus_init(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """K-Means++ centroid initialization.

        Pick the first centroid uniformly at random, then each subsequent
        centroid with probability proportional to D(x)^2 — the squared
        distance from x to the nearest already-chosen centroid.  This
        reduces expected inertia and the number of iterations to convergence
        compared to plain random initialization.
        """
        n_samples = X.shape[0]
        first_idx = int(rng.integers(0, n_samples))
        centroids = [X[first_idx].copy()]

        for _ in range(1, self.n_clusters):
            # Squared distance from each point to its nearest centroid so far.
            centroid_matrix = np.array(centroids)          # (k_so_far, n_features)
            diffs = X[:, np.newaxis, :] - centroid_matrix  # (n, k_so_far, n_features)
            sq_dists = (diffs ** 2).sum(axis=2)            # (n, k_so_far)
            min_sq_dists = sq_dists.min(axis=1)            # (n,)  D(x)^2

            # Sample next centroid proportional to D(x)^2.
            total = min_sq_dists.sum()
            if total == 0:
                # All remaining points coincide with existing centroids; fall back.
                probs = np.ones(n_samples) / n_samples
            else:
                probs = min_sq_dists / total
            next_idx = int(rng.choice(n_samples, p=probs))
            centroids.append(X[next_idx].copy())

        return np.array(centroids)

    # ── Single run ────────────────────────────────────────────────────────────

    def _fit_once(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """One full K-Means run from a fresh K-Means++ initialization."""
        centroids = self._kmeans_plus_plus_init(X, rng)

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X, centroids)
            new_centroids = np.zeros_like(centroids)

            for k in range(self.n_clusters):
                pts = X[labels == k]
                # If cluster is empty, reinitialize to a random data point.
                new_centroids[k] = (
                    X[rng.integers(0, X.shape[0])] if len(pts) == 0 else pts.mean(axis=0)
                )

            # A3: max per-centroid shift, not Frobenius norm over all centroids.
            # Standard convergence criterion: max_k ||c_k_new − c_k_old||.
            shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
            centroids = new_centroids

            if shift < self.tol:
                break

        labels = self._assign_clusters(X, centroids)
        inertia = self._compute_inertia(X, labels, centroids)
        return centroids, labels, inertia

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        # A4: run n_init independent restarts; keep the lowest-inertia result.
        best_centroids, best_labels, best_inertia = None, None, float("inf")
        for _ in range(self.n_init):
            centroids, labels, inertia = self._fit_once(X, rng)
            if inertia < best_inertia:
                best_centroids, best_labels, best_inertia = centroids, labels, inertia

        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.centroids is None:
            raise ValueError("Model hasn't been fitted yet")
        return self._assign_clusters(X, self.centroids)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    # ── Internals ─────────────────────────────────────────────────────────────

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        # (n_samples, n_clusters, n_features) pairwise distances
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_inertia(
        self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
    ) -> float:
        total = 0.0
        for k in range(self.n_clusters):
            pts = X[labels == k]
            if len(pts) > 0:
                total += float(np.sum((pts - centroids[k]) ** 2))
        return total
