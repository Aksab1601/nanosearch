"""IVF: skip most of the database by partitioning it first.

Exact search looks at all ``n`` vectors. An inverted-file index clusters the
vectors into ``nlist`` cells once, at build time, and at query time scores only
the ``nprobe`` cells whose centroids are closest to the query -- about
``nprobe / nlist`` of the corpus.

Cost per query goes from ``O(n * d)`` to roughly::

    O(nlist * d)            find the nearest centroids
  + O(n * nprobe/nlist * d) score the candidates in those cells

With ``nlist ~ sqrt(n)`` the two terms balance, which is where the usual
"nlist = sqrt(n) to 4*sqrt(n)" rule of thumb comes from.

The approximation is specific and worth stating precisely: a true neighbour
sitting in a cell that was not probed can never be returned. That is the only
error IVF makes, and it is why recall climbs as ``nprobe`` does -- ``nprobe =
nlist`` reduces exactly to brute force. ``nprobe`` is therefore a *runtime*
accuracy dial on an already-built index, which is unusually convenient: the
same index can serve a cheap autocomplete path and an expensive
high-recall path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseIndex, Metric, SearchResult
from .inverted import InvertedLists
from .kmeans import KMeans
from .metrics import FloatArray, IntArray, squared_l2, squared_norms, topk_smallest

__all__ = ["IVFIndex"]


class IVFIndex(BaseIndex):
    """Inverted-file index over full-precision vectors.

    Parameters
    ----------
    nlist:
        Number of cells (coarse centroids).
    nprobe:
        Default number of cells to visit per query. Overridable per call.
    """

    def __init__(
        self,
        dim: int,
        nlist: int = 256,
        metric: Metric = "cosine",
        nprobe: int = 8,
        kmeans_iter: int = 25,
        seed: int = 0,
        block_size: int = 8192,
    ) -> None:
        super().__init__(dim, metric)
        if nprobe <= 0:
            raise ValueError(f"nprobe must be positive, got {nprobe}")
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.kmeans_iter = int(kmeans_iter)
        self.seed = int(seed)
        self.block_size = int(block_size)

        self.centroids_: FloatArray = np.zeros((0, self.dim), dtype=np.float32)
        self._centroid_sqnorms: FloatArray = np.zeros(0, dtype=np.float32)
        self._lists = InvertedLists(self.nlist)
        self._data: FloatArray = np.zeros((0, self.dim), dtype=np.float32)
        self._sqnorms: FloatArray = np.zeros(0, dtype=np.float32)
        self._size = 0

    # ------------------------------------------------------------ properties

    @property
    def is_trained(self) -> bool:
        return self.centroids_.shape[0] == self.nlist

    @property
    def vectors(self) -> FloatArray:
        return self._data[: self._size]

    def clone_trained(self) -> IVFIndex:
        """An empty index sharing this one's learned centroids (see IVFPQIndex)."""
        clone = IVFIndex(
            dim=self.dim,
            nlist=self.nlist,
            metric=self.metric,
            nprobe=self.nprobe,
            kmeans_iter=self.kmeans_iter,
            seed=self.seed,
            block_size=self.block_size,
        )
        clone.centroids_ = self.centroids_.copy()
        clone._centroid_sqnorms = self._centroid_sqnorms.copy()
        return clone

    # ----------------------------------------------------------------- train

    def train(self, vectors: Any) -> None:
        """Learn the ``nlist`` coarse centroids.

        Training is separate from ``add`` because the partition must be fixed
        before anything is inserted -- a vector's cell is part of how it is
        stored. In production the usual pattern is to train on a sample (a few
        hundred vectors per cell is plenty) and then stream the full corpus
        through ``add``.
        """
        x = self._prepare(vectors)
        if x.shape[0] < self.nlist:
            raise ValueError(f"need at least nlist={self.nlist} training vectors, got {x.shape[0]}")
        km = KMeans(n_clusters=self.nlist, n_iter=self.kmeans_iter, seed=self.seed).fit(x)
        self.centroids_ = km.centroids_
        self._centroid_sqnorms = squared_norms(self.centroids_)

    # ------------------------------------------------------------------- add

    def _ensure_capacity(self, extra: int) -> None:
        needed = self._size + extra
        capacity = self._data.shape[0]
        if needed <= capacity:
            return
        new_capacity = max(1024, needed, capacity * 2)
        grown = np.zeros((new_capacity, self.dim), dtype=np.float32)
        grown[: self._size] = self._data[: self._size]
        self._data = grown
        grown_norms = np.zeros(new_capacity, dtype=np.float32)
        grown_norms[: self._size] = self._sqnorms[: self._size]
        self._sqnorms = grown_norms

    def add(self, vectors: Any, ids: Any | None = None) -> IntArray:
        if not self.is_trained:
            raise RuntimeError("IVFIndex must be trained before vectors are added")
        prepared = self._prepare(vectors)
        n = prepared.shape[0]
        assigned = self._next_ids(n, ids)

        labels = self._assign_cells(prepared)
        positions = np.arange(self._size, self._size + n, dtype=np.int64)

        self._ensure_capacity(n)
        self._data[self._size : self._size + n] = prepared
        self._sqnorms[self._size : self._size + n] = squared_norms(prepared)
        self._size += n
        self._ids = np.concatenate([self._ids, assigned])
        self._lists.extend(labels, positions)
        return assigned

    def _assign_cells(self, x: FloatArray) -> IntArray:
        labels = np.empty(x.shape[0], dtype=np.int64)
        for start in range(0, x.shape[0], self.block_size):
            stop = min(start + self.block_size, x.shape[0])
            dist = squared_l2(x[start:stop], self.centroids_, vector_sqnorms=self._centroid_sqnorms)
            labels[start:stop] = np.argmin(dist, axis=1)
        return labels

    # ---------------------------------------------------------------- search

    def search(
        self, queries: Any, k: int = 10, nprobe: int | None = None, **_unused: Any
    ) -> SearchResult:
        if not self.is_trained:
            raise RuntimeError("IVFIndex must be trained before searching")
        q = self._prepare(queries)
        nq = q.shape[0]
        probe = int(nprobe or self.nprobe)
        probe = max(1, min(probe, self.nlist))

        ids, scores = self._blank(nq, k)
        if self._size == 0:
            return SearchResult(ids, scores)

        # Stage 1: which cells to visit. One (nq, nlist) matrix multiply.
        coarse = squared_l2(q, self.centroids_, vector_sqnorms=self._centroid_sqnorms)
        probe_lists, _ = topk_smallest(coarse, probe)

        # Stage 2: exact distances to the candidates in those cells.
        for i in range(nq):
            candidates = self._lists.gather(probe_lists[i])
            if candidates.size == 0:
                continue
            dist = squared_l2(
                q[i : i + 1],
                self._data[candidates],
                vector_sqnorms=self._sqnorms[candidates],
            )[0]
            take = min(k, candidates.shape[0])
            sel, vals = topk_smallest(dist, take)
            ids[i, :take] = self._ids[candidates[sel]]
            scores[i, :take] = self._to_score(vals)

        return SearchResult(ids, scores)

    # ----------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        sizes = self._lists.sizes()
        vector_bytes = self._size * self.dim * 4
        centroid_bytes = self.centroids_.size * 4
        return {
            "index_type": "ivf",
            "metric": self.metric,
            "dim": self.dim,
            "ntotal": self._size,
            "is_trained": self.is_trained,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "bytes_per_vector": self.dim * 4,
            "memory_mb": round((vector_bytes + centroid_bytes) / 1e6, 3),
            "list_size_min": int(sizes.min()) if sizes.size else 0,
            "list_size_max": int(sizes.max()) if sizes.size else 0,
            "list_size_mean": round(float(sizes.mean()), 2) if sizes.size else 0.0,
            "imbalance_factor": round(self._lists.imbalance_factor(), 4),
            "exact": False,
        }

    # ----------------------------------------------------------- persistence

    def _state(self) -> dict[str, Any]:
        offsets, flat = self._lists.to_csr()
        return {
            "vectors": self.vectors.copy(),
            "ids": self._ids,
            "centroids": self.centroids_,
            "list_offsets": offsets,
            "list_entries": flat,
            "dim": self.dim,
            "metric": self.metric,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "kmeans_iter": self.kmeans_iter,
            "seed": self.seed,
            "block_size": self.block_size,
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> IVFIndex:
        index = cls(
            dim=int(state["dim"]),
            nlist=int(state["nlist"]),
            metric=str(state["metric"]),
            nprobe=int(state["nprobe"]),
            kmeans_iter=int(state["kmeans_iter"]),
            seed=int(state["seed"]),
            block_size=int(state["block_size"]),
        )
        index.centroids_ = np.asarray(state["centroids"], dtype=np.float32)
        index._centroid_sqnorms = squared_norms(index.centroids_)
        vectors = np.asarray(state["vectors"], dtype=np.float32)
        index._ensure_capacity(vectors.shape[0])
        index._data[: vectors.shape[0]] = vectors
        index._sqnorms[: vectors.shape[0]] = squared_norms(vectors)
        index._size = vectors.shape[0]
        index._ids = np.asarray(state["ids"], dtype=np.int64)
        index._lists = InvertedLists.from_csr(state["list_offsets"], state["list_entries"])
        return index
