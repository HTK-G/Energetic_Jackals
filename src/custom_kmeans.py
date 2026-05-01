import numpy as np


class CustomKMeans:
    def __init__(
        self,
        n_clusters: int,
        max_iters: int = 100,
        tol: float = 1e-4,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol

        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.random_state = random_state

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

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        # K-Means++ initialization
        self.centroids = self._kmeans_plus_plus_init(X, rng)

        for i in range(self.max_iters):
            labels = self._assign_clusters(X)
            new_centroids = np.zeros_like(self.centroids)

            for clusteridx in range(self.n_clusters):
                clusterpoints = X[labels == clusteridx]
                # if cluster is empty, reinizialize to random data point
                if len(clusterpoints) == 0:
                    new_centroids[clusteridx] = X[rng.integers(0, X.shape[0])]
                else:
                    new_centroids[clusteridx] = np.mean(clusterpoints, axis=0)

            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if centroid_shift < self.tol:
                break
        self.labels_ = self._assign_clusters(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.centroids is None:
            raise ValueError("Model hasn't been fitted yet")
        return self._assign_clusters(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def _assign_clusters(self, X):
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.centroids, axis=2
        )  # shape of x[:,np.newaxis] - self.centroids is (n_samples, k, n_features)
        # gets the pairwise distances of each point and each cluster. computes distances across features
        return np.argmin(distances, axis=1)

    def _compute_inertia(self, X, labels):
        # sum of squared distances between each datapoint and its centroid
        total = 0.0
        for clusteridx in range(self.n_clusters):
            clusterpoints = X[labels == clusteridx]
            if len(clusterpoints) > 0:
                total += np.sum((clusterpoints - self.centroids[clusteridx]) ** 2)
        return total
