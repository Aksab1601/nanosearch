"""Exact (brute-force) search -- the baseline every approximate index is judged against.

``FlatIndex`` compares the query against every stored vector, so its results
are the ground truth by definition: recall@k for the approximate indexes is
measured against this. It is also genuinely the right choice below roughly
10k-50k vectors, where one BLAS matrix multiply beats the overhead of
maintaining a partitioned index.

Cost: O(n * d) per query, with the constant factor set by BLAS rather than by
Python.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseIndex, Metric, SearchResult
from .metrics import FloatArray, IntArray, squared_l2, squared_norms, topk_smallest

__all__ = ["FlatIndex"]


class FlatIndex(BaseIndex):
    """Exhaustive search over float32 vectors.

    Parameters
    ----------
    block_size:
        Database rows scored per pass. The distance matrix for one pass is
        ``(n_queries, block_size)``; without blocking, a 1000-query batch
        against 10M vectors would try to allocate 40 GB. Blocking trades a
        handful of extra BLAS calls for bounded memory.
    """

    def __init__(self, dim: int, metric: Metric = "cosine", block_size: int = 8192) -> None:
        super().__init__(dim, metric)
        self.block_size = int(block_size)
        self._data: FloatArray = np.zeros((0, self.dim), dtype=np.float32)
        self._sqnorms: FloatArray = np.zeros(0, dtype=np.float32)
        self._size = 0

    # --------------------------------------------------------------- storage

    @property
    def vectors(self) -> FloatArray:
        """A view of the stored vectors (post-normalisation for cosine)."""
        return self._data[: self._size]

    def _ensure_capacity(self, extra: int) -> None:
        """Grow the backing store geometrically.

        Doubling keeps the amortised cost of an append O(1) -- the same
        argument behind Python's own list and C++'s ``std::vector``. Growing by
        a fixed amount instead would make repeated ``add`` calls O(n^2)
        overall.
        """
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

    # ------------------------------------------------------------------- add

    def add(self, vectors: Any, ids: Any | None = None) -> IntArray:
        prepared = self._prepare(vectors)
        n = prepared.shape[0]
        assigned = self._next_ids(n, ids)

        self._ensure_capacity(n)
        self._data[self._size : self._size + n] = prepared
        # Cached ||x||^2: the database does not change between queries, so
        # paying for these once at insert time removes them from the hot path.
        self._sqnorms[self._size : self._size + n] = squared_norms(prepared)
        self._size += n
        self._ids = np.concatenate([self._ids, assigned])
        return assigned

    # ---------------------------------------------------------------- search

    def search(self, queries: Any, k: int = 10, **_unused: Any) -> SearchResult:
        q = self._prepare(queries)
        nq = q.shape[0]
        if self._size == 0:
            return SearchResult(
                np.full((nq, k), -1, dtype=np.int64),
                np.full((nq, k), np.inf if self.metric == "l2" else -np.inf, dtype=np.float32),
            )

        k_eff = min(k, self._size)
        best_pos = np.full((nq, 0), 0, dtype=np.int64)
        best_dist = np.full((nq, 0), 0.0, dtype=np.float32)

        for start in range(0, self._size, self.block_size):
            stop = min(start + self.block_size, self._size)
            dist = squared_l2(q, self._data[start:stop], vector_sqnorms=self._sqnorms[start:stop])
            idx, vals = topk_smallest(dist, k_eff)
            # Merge this block's best with the running best, keeping k_eff.
            cand_pos = np.concatenate([best_pos, idx + start], axis=1)
            cand_dist = np.concatenate([best_dist, vals], axis=1)
            sel, best_dist = topk_smallest(cand_dist, k_eff)
            best_pos = np.take_along_axis(cand_pos, sel, axis=1)

        return self._finalise(best_pos, best_dist, nq, k)

    def _finalise(self, positions: IntArray, sqdist: FloatArray, nq: int, k: int) -> SearchResult:
        ids = np.full((nq, k), -1, dtype=np.int64)
        scores = np.full((nq, k), -np.inf if self.metric == "cosine" else np.inf, np.float32)
        found = positions.shape[1]
        ids[:, :found] = self._ids[positions]
        scores[:, :found] = self._to_score(sqdist)
        return SearchResult(ids, scores)

    # ----------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        vector_bytes = self._size * self.dim * 4
        return {
            "index_type": "flat",
            "metric": self.metric,
            "dim": self.dim,
            "ntotal": self._size,
            "is_trained": True,
            "bytes_per_vector": self.dim * 4,
            "memory_mb": round(vector_bytes / 1e6, 3),
            "exact": True,
        }

    # ----------------------------------------------------------- persistence

    def _state(self) -> dict[str, Any]:
        return {
            "vectors": self.vectors.copy(),
            "ids": self._ids,
            "dim": self.dim,
            "metric": self.metric,
            "block_size": self.block_size,
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> FlatIndex:
        index = cls(
            dim=int(state["dim"]),
            metric=str(state["metric"]),
            block_size=int(state["block_size"]),
        )
        vectors = np.asarray(state["vectors"], dtype=np.float32)
        index._ensure_capacity(vectors.shape[0])
        index._data[: vectors.shape[0]] = vectors
        index._sqnorms[: vectors.shape[0]] = squared_norms(vectors)
        index._size = vectors.shape[0]
        index._ids = np.asarray(state["ids"], dtype=np.int64)
        return index
