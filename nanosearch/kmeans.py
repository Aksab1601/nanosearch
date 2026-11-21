"""Lloyd's k-means with k-means++ initialisation, in NumPy.

k-means appears twice in this project, doing two unrelated jobs:

* the **coarse quantiser** of the IVF index partitions the vector space into
  ``nlist`` cells, so a query only has to look inside a few of them
* each **sub-quantiser** of the product quantiser learns a 256-entry codebook
  for one slice of the vector, so a vector can be stored as bytes

Both are the same algorithm, which is why it lives in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import FloatArray, IntArray, as_float32, squared_l2, squared_norms

__all__ = ["KMeans"]


@dataclass
class KMeans:
    """Minimal, vectorised k-means.

    Parameters
    ----------
    n_clusters:
        Number of centroids to learn.
    n_iter:
        Maximum Lloyd iterations.
    tol:
        Stop early when the relative drop in inertia between iterations falls
        below this value. Clustering for an index does not need to run to
        convergence: a slightly worse partition costs a little recall, while
        extra iterations cost build time on every single index build.
    seed:
        Seed for the RNG, so index builds are reproducible.
    max_points_per_centroid:
        Training vectors are subsampled to at most
        ``n_clusters * max_points_per_centroid`` points. Centroid positions
        stop improving long before every point is used, so this caps build
        time on large corpora with almost no quality loss. (FAISS applies the
        same trick with the same default.)
    kmeanspp_sample:
        k-means++ costs O(n * k * d), which dominates everything else when k is
        large, so the seeding scan runs on at most this many points.
    chunk_size:
        Rows of the assignment step processed at a time, bounding the
        ``(chunk, n_clusters)`` distance matrix rather than materialising an
        ``(n, n_clusters)`` one.
    """

    n_clusters: int
    n_iter: int = 25
    tol: float = 1e-4
    seed: int = 0
    max_points_per_centroid: int = 256
    kmeanspp_sample: int = 20_000
    chunk_size: int = 8192

    centroids_: FloatArray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    inertia_: float = float("inf")
    n_iter_: int = 0

    # ------------------------------------------------------------------ fit

    def fit(self, x: FloatArray) -> KMeans:
        """Learn ``n_clusters`` centroids from ``x``."""
        data = as_float32(x)
        n, d = data.shape
        if n == 0:
            raise ValueError("cannot fit k-means on an empty array")
        if self.n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {self.n_clusters}")

        rng = np.random.default_rng(self.seed)
        train = self._subsample(data, self.n_clusters * self.max_points_per_centroid, rng)

        if train.shape[0] <= self.n_clusters:
            # Degenerate but legal: fewer training points than centroids. Use
            # every point as a centroid and pad by repeating points, so the
            # codebook still has the promised shape.
            reps = int(np.ceil(self.n_clusters / train.shape[0]))
            centroids = np.tile(train, (reps, 1))[: self.n_clusters].copy()
            self.centroids_ = centroids
            labels, dist = self._assign(train, centroids)
            self.inertia_ = float(dist.sum())
            self.n_iter_ = 0
            return self

        centroids = self._kmeanspp_init(train, rng)
        prev_inertia = float("inf")

        for iteration in range(1, self.n_iter + 1):
            labels, min_dist = self._assign(train, centroids)
            inertia = float(min_dist.sum())
            centroids = self._update(train, labels, centroids, min_dist, rng)
            self.n_iter_ = iteration
            if prev_inertia < float("inf"):
                improvement = (prev_inertia - inertia) / max(prev_inertia, 1e-12)
                if improvement < self.tol:
                    prev_inertia = inertia
                    break
            prev_inertia = inertia

        self.centroids_ = np.ascontiguousarray(centroids, dtype=np.float32)
        self.inertia_ = prev_inertia
        return self

    # --------------------------------------------------------------- assign

    def assign(self, x: FloatArray) -> IntArray:
        """Return the index of the nearest centroid for each row of ``x``."""
        self._check_fitted()
        labels, _ = self._assign(as_float32(x), self.centroids_)
        return labels

    def assign_with_distance(self, x: FloatArray) -> tuple[IntArray, FloatArray]:
        """Nearest centroid and its squared distance, for each row of ``x``."""
        self._check_fitted()
        return self._assign(as_float32(x), self.centroids_)

    # -------------------------------------------------------------- helpers

    def _check_fitted(self) -> None:
        if self.centroids_.size == 0:
            raise RuntimeError("KMeans is not fitted; call fit() first")

    @staticmethod
    def _subsample(x: FloatArray, limit: int, rng: np.random.Generator) -> FloatArray:
        if x.shape[0] <= limit:
            return x
        picks = rng.choice(x.shape[0], size=limit, replace=False)
        return np.ascontiguousarray(x[np.sort(picks)])

    def _kmeanspp_init(self, x: FloatArray, rng: np.random.Generator) -> FloatArray:
        """k-means++ seeding: pick centres far away from the ones chosen so far.

        Each new centre is sampled with probability proportional to its squared
        distance to the nearest existing centre. Uniform seeding frequently
        drops several centres into the same dense region, which wastes
        capacity; k-means++ spreads them out and converges in fewer
        iterations.
        """
        pool = self._subsample(x, max(self.kmeanspp_sample, self.n_clusters), rng)
        n, d = pool.shape
        centroids = np.empty((self.n_clusters, d), dtype=np.float32)

        first = int(rng.integers(n))
        centroids[0] = pool[first]
        closest = squared_l2(centroids[0:1], pool)[0]

        for i in range(1, self.n_clusters):
            total = float(closest.sum())
            if total <= 0.0:
                # All remaining points coincide with a chosen centre.
                centroids[i:] = pool[rng.integers(n, size=self.n_clusters - i)]
                break
            # Sample with probability proportional to squared distance by
            # inverse-CDF: cumsum then binary search. np.random.choice(p=...)
            # is the obvious call here but builds and validates an n-length
            # probability vector on every one of the k iterations, which
            # dominated PQ training time (k=256 per subspace, 16 subspaces).
            cdf = np.cumsum(closest, dtype=np.float64)
            pick = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
            pick = min(pick, n - 1)
            centroids[i] = pool[pick]
            np.minimum(closest, squared_l2(centroids[i : i + 1], pool)[0], out=closest)

        return centroids

    def _assign(self, x: FloatArray, centroids: FloatArray) -> tuple[IntArray, FloatArray]:
        n = x.shape[0]
        labels = np.empty(n, dtype=np.int64)
        min_dist = np.empty(n, dtype=np.float32)
        c_sq = squared_norms(centroids)

        for start in range(0, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            dist = squared_l2(x[start:stop], centroids, vector_sqnorms=c_sq)
            chunk_labels = np.argmin(dist, axis=1)
            labels[start:stop] = chunk_labels
            min_dist[start:stop] = dist[np.arange(stop - start), chunk_labels]

        return labels, min_dist

    def _update(
        self,
        x: FloatArray,
        labels: IntArray,
        centroids: FloatArray,
        min_dist: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Move each centroid to the mean of its members.

        The per-cluster sum is a scatter-add. ``np.bincount`` with ``weights``
        does exactly that in C, one dimension at a time -- faster than
        ``np.add.at`` (which is unbuffered and slow) and without the memory
        spike of building an (n, d) prefix-sum array.
        """
        k, d = centroids.shape
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        sums = np.empty((k, d), dtype=np.float32)
        for j in range(d):
            sums[:, j] = np.bincount(labels, weights=x[:, j], minlength=k)

        empty = counts == 0
        new_centroids = np.empty_like(centroids)
        nonempty = ~empty
        new_centroids[nonempty] = sums[nonempty] / counts[nonempty, None]

        if empty.any():
            # A centroid nobody claimed is dead weight: it will never win an
            # assignment again. Respawn it on the worst-served point, which
            # splits the cluster that is currently costing the most inertia.
            n_empty = int(empty.sum())
            worst = np.argpartition(-min_dist, min(n_empty, min_dist.shape[0] - 1))[:n_empty]
            jitter = rng.normal(0.0, 1e-4, size=(n_empty, d)).astype(np.float32)
            new_centroids[empty] = x[worst] + jitter

        return new_centroids
