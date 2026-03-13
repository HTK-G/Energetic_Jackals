"""A NumPy implementation of K-Means clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class KMeansResult:
    """Summary of a fitted K-Means model."""

    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    iterations: int


class NumpyKMeans:
    """K-Means clustering from scratch without relying on sklearn's implementation."""

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 100,
        tol: float = 1e-4,
        init: str = "kmeans++",
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0

    def _squared_distances(self, samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return np.sum((samples[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    def _initialize_centroids(self, samples: np.ndarray) -> np.ndarray:
        if self.n_clusters > len(samples):
            raise ValueError("n_clusters cannot exceed the number of samples.")

        rng = np.random.default_rng(self.random_state)
        first_center_index = int(rng.integers(0, len(samples)))
        centers = [samples[first_center_index]]

        if self.init == "random":
            remaining_indices = rng.choice(len(samples), size=self.n_clusters - 1, replace=False)
            centers.extend(samples[index] for index in remaining_indices)
            return np.asarray(centers, dtype=float)

        while len(centers) < self.n_clusters:
            distance_to_nearest_center = self._squared_distances(samples, np.asarray(centers)).min(axis=1)
            total_distance = distance_to_nearest_center.sum()

            if total_distance == 0.0:
                candidate_index = int(rng.integers(0, len(samples)))
            else:
                probabilities = distance_to_nearest_center / total_distance
                candidate_index = int(rng.choice(len(samples), p=probabilities))
            centers.append(samples[candidate_index])

        return np.asarray(centers, dtype=float)

    def _repair_empty_clusters(self, samples: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
        repaired_centers = centers.copy()
        distances = self._squared_distances(samples, repaired_centers)
        nearest_distances = distances.min(axis=1)

        for cluster_index in range(self.n_clusters):
            if np.any(labels == cluster_index):
                continue
            replacement_index = int(np.argmax(nearest_distances))
            repaired_centers[cluster_index] = samples[replacement_index]
            nearest_distances[replacement_index] = -np.inf

        return repaired_centers

    def fit(self, samples: np.ndarray) -> "NumpyKMeans":
        samples = np.asarray(samples, dtype=float)
        centers = self._initialize_centroids(samples)

        for iteration in range(1, self.max_iter + 1):
            distances = self._squared_distances(samples, centers)
            labels = distances.argmin(axis=1)

            updated_centers = centers.copy()
            for cluster_index in range(self.n_clusters):
                cluster_samples = samples[labels == cluster_index]
                if len(cluster_samples) > 0:
                    updated_centers[cluster_index] = cluster_samples.mean(axis=0)

            updated_centers = self._repair_empty_clusters(samples, labels, updated_centers)
            centroid_shift = float(np.linalg.norm(updated_centers - centers))
            centers = updated_centers

            if centroid_shift <= self.tol:
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter

        final_distances = self._squared_distances(samples, centers)
        final_labels = final_distances.argmin(axis=1)
        final_inertia = float(np.sum((samples - centers[final_labels]) ** 2))

        self.cluster_centers_ = centers
        self.labels_ = final_labels
        self.inertia_ = final_inertia
        return self

    def fit_predict(self, samples: np.ndarray) -> np.ndarray:
        self.fit(samples)
        return np.asarray(self.labels_, dtype=int)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("NumpyKMeans must be fitted before calling predict().")
        samples = np.asarray(samples, dtype=float)
        distances = self._squared_distances(samples, self.cluster_centers_)
        return distances.argmin(axis=1)

    def to_result(self) -> KMeansResult:
        if self.cluster_centers_ is None or self.labels_ is None or self.inertia_ is None:
            raise RuntimeError("NumpyKMeans must be fitted before requesting results.")
        return KMeansResult(
            labels=self.labels_,
            centroids=self.cluster_centers_,
            inertia=self.inertia_,
            iterations=self.n_iter_,
        )
