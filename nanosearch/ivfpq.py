"""IVF-PQ: the index that makes large-scale vector search practical.

This combines the two ideas in the rest of the package:

* **IVF** cuts how *many* vectors get scored (partition, probe a few cells)
* **PQ** cuts how *expensive* scoring each one is, and how much memory it takes

Residual encoding
-----------------
PQ is applied to ``x - centroid(x)``, not to ``x``. Residuals are much smaller
and more tightly concentrated than the raw vectors, so the same 256-entry
codebooks resolve them far more finely -- the coarse quantiser has already
captured where in the space the vector lives, leaving PQ to describe only the
local offset. This is a free accuracy win at identical storage cost, and it is
why real systems always do it.

It works because the residual is taken against the *same* centroid on both
sides. For a query ``q`` and a vector ``x`` in the cell with centroid ``c``::

    ||q - x||^2 = ||(q - c) - (x - c)||^2

so a query is compared with a cell's codes by first forming its own residual
against that cell's centroid. Distances from different cells remain directly
comparable, because each one approximates the same quantity ``||q - x||^2``.

Memory, for 1M vectors at 768-d
-------------------------------
* float32: ``1e6 * 768 * 4`` = **3.07 GB**
* IVF-PQ, m=48: ``1e6 * 48`` = **48 MB** plus codebooks and centroids

That is the difference between a cluster of machines and one container, and it
is the reason production systems accept a few points of recall.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseIndex, Metric, SearchResult
from .inverted import InvertedLists
from .kmeans import KMeans
from .metrics import FloatArray, IntArray, squared_l2, squared_norms, topk_smallest
from .pq import ProductQuantizer

__all__ = ["IVFPQIndex"]


class IVFPQIndex(BaseIndex):
    """Inverted file index over product-quantised residuals.

    Parameters
    ----------
    nlist:
        Number of cells. Rule of thumb: ``sqrt(n)`` to ``4 * sqrt(n)``.
    m:
        Sub-vectors per code, i.e. bytes per stored vector. Must divide ``dim``.
    nprobe:
        Default cells probed per query -- the runtime recall dial.
    """

    def __init__(
        self,
        dim: int,
        nlist: int = 256,
        m: int = 8,
        nbits: int = 8,
        metric: Metric = "cosine",
        nprobe: int = 8,
        kmeans_iter: int = 25,
        pq_iter: int = 25,
        seed: int = 0,
        block_size: int = 8192,
        pq_train_size: int = 100_000,
    ) -> None:
        super().__init__(dim, metric)
        if nprobe <= 0:
            raise ValueError(f"nprobe must be positive, got {nprobe}")
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.kmeans_iter = int(kmeans_iter)
        self.seed = int(seed)
        self.block_size = int(block_size)
        self.pq_train_size = int(pq_train_size)

        self._pq = ProductQuantizer(dim=dim, m=m, nbits=nbits, n_iter=pq_iter, seed=seed + 1)
        self.centroids_: FloatArray = np.zeros((0, self.dim), dtype=np.float32)
        self._centroid_sqnorms: FloatArray = np.zeros(0, dtype=np.float32)
        self._lists = InvertedLists(self.nlist)
        self._codes: np.ndarray = np.zeros((0, self._pq.m), dtype=np.uint8)
        self._size = 0

    # ------------------------------------------------------------ properties

    @property
    def m(self) -> int:
        return self._pq.m

    @property
    def is_trained(self) -> bool:
        return self.centroids_.shape[0] == self.nlist and self._pq.is_trained

    @property
    def codes(self) -> np.ndarray:
        return self._codes[: self._size]

    def clone_trained(self) -> IVFPQIndex:
        """An empty index sharing this one's learned centroids and codebooks.

        Training is the expensive part of building an index and it depends only
        on the *distribution*, not on the particular vectors stored. So a
        trained quantiser can be reused: this is how sharding works in a
        distributed vector store -- train once on a sample, clone to every
        shard, and each shard indexes its own slice. Because the shards share
        codebooks, their scores are directly comparable and a top-k merge
        across shards is just another top-k.
        """
        clone = IVFPQIndex(
            dim=self.dim,
            nlist=self.nlist,
            m=self.m,
            nbits=self._pq.nbits,
            metric=self.metric,
            nprobe=self.nprobe,
            kmeans_iter=self.kmeans_iter,
            pq_iter=self._pq.n_iter,
            seed=self.seed,
            block_size=self.block_size,
            pq_train_size=self.pq_train_size,
        )
        clone.centroids_ = self.centroids_.copy()
        clone._centroid_sqnorms = self._centroid_sqnorms.copy()
        clone._pq.codebooks_ = self._pq.codebooks_.copy()
        return clone

    # ----------------------------------------------------------------- train

    def train(self, vectors: Any) -> None:
        """Learn the coarse centroids, then the PQ codebooks on the residuals.

        Order matters: the residuals the quantiser must represent only exist
        once the partition is fixed.
        """
        x = self._prepare(vectors)
        if x.shape[0] < self.nlist:
            raise ValueError(f"need at least nlist={self.nlist} training vectors, got {x.shape[0]}")

        coarse = KMeans(n_clusters=self.nlist, n_iter=self.kmeans_iter, seed=self.seed).fit(x)
        self.centroids_ = coarse.centroids_
        self._centroid_sqnorms = squared_norms(self.centroids_)

        # Codebooks are fitted on a sample of the residuals. 256 centroids per
        # subspace want a few tens of thousands of points to be stable, not
        # millions -- and the alternative is materialising a residual array the
        # size of the whole corpus just to throw it away.
        sample = x
        if x.shape[0] > self.pq_train_size:
            rng = np.random.default_rng(self.seed + 7)
            picks = np.sort(rng.choice(x.shape[0], size=self.pq_train_size, replace=False))
            sample = np.ascontiguousarray(x[picks])
        labels = self._assign_cells(sample)
        self._pq.fit(sample - self.centroids_[labels])

    # ------------------------------------------------------------------- add

    def _assign_cells(self, x: FloatArray) -> IntArray:
        labels = np.empty(x.shape[0], dtype=np.int64)
        for start in range(0, x.shape[0], self.block_size):
            stop = min(start + self.block_size, x.shape[0])
            dist = squared_l2(x[start:stop], self.centroids_, vector_sqnorms=self._centroid_sqnorms)
            labels[start:stop] = np.argmin(dist, axis=1)
        return labels

    def _ensure_capacity(self, extra: int) -> None:
        needed = self._size + extra
        capacity = self._codes.shape[0]
        if needed <= capacity:
            return
        new_capacity = max(1024, needed, capacity * 2)
        grown = np.zeros((new_capacity, self.m), dtype=np.uint8)
        grown[: self._size] = self._codes[: self._size]
        self._codes = grown

    def add(self, vectors: Any, ids: Any | None = None) -> IntArray:
        if not self.is_trained:
            raise RuntimeError("IVFPQIndex must be trained before vectors are added")
        prepared = self._prepare(vectors)
        n = prepared.shape[0]
        assigned = self._next_ids(n, ids)

        labels = self._assign_cells(prepared)
        # Residuals are formed and encoded a block at a time so that a bulk
        # load never allocates a second copy of the whole batch.
        codes = np.empty((n, self.m), dtype=np.uint8)
        for start in range(0, n, self.block_size):
            stop = min(start + self.block_size, n)
            block = prepared[start:stop] - self.centroids_[labels[start:stop]]
            codes[start:stop] = self._pq.encode(block)

        positions = np.arange(self._size, self._size + n, dtype=np.int64)
        self._ensure_capacity(n)
        self._codes[self._size : self._size + n] = codes
        self._size += n
        self._ids = np.concatenate([self._ids, assigned])
        self._lists.extend(labels, positions)
        # The full-precision vectors are deliberately not retained: dropping
        # them is the entire memory win.
        return assigned

    # ---------------------------------------------------------------- search

    def search(
        self, queries: Any, k: int = 10, nprobe: int | None = None, **_unused: Any
    ) -> SearchResult:
        if not self.is_trained:
            raise RuntimeError("IVFPQIndex must be trained before searching")
        q = self._prepare(queries)
        nq = q.shape[0]
        probe = max(1, min(int(nprobe or self.nprobe), self.nlist))

        ids, scores = self._blank(nq, k)
        if self._size == 0:
            return SearchResult(ids, scores)

        coarse = squared_l2(q, self.centroids_, vector_sqnorms=self._centroid_sqnorms)
        probe_lists, _ = topk_smallest(coarse, probe)

        for i in range(nq):
            chosen = probe_lists[i]
            chunks = [self._lists[int(list_id)] for list_id in chosen]
            sizes = np.array([c.shape[0] for c in chunks], dtype=np.int64)
            total = int(sizes.sum())
            if total == 0:
                continue

            candidates = np.concatenate([c for c in chunks if c.size])
            # Which probed cell each candidate came from, so it can be scored
            # against that cell's residual table.
            groups = np.repeat(np.arange(chosen.shape[0], dtype=np.int64), sizes)

            residual_queries = q[i] - self.centroids_[chosen]
            tables = self._pq.distance_tables(residual_queries)
            dist = self._pq.asymmetric_distances_grouped(tables, self._codes[candidates], groups)

            take = min(k, total)
            sel, vals = topk_smallest(dist, take)
            ids[i, :take] = self._ids[candidates[sel]]
            scores[i, :take] = self._to_score(vals)

        return SearchResult(ids, scores)

    # ----------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        sizes = self._lists.sizes()
        code_bytes = self._size * self.m
        centroid_bytes = self.centroids_.size * 4
        codebook_bytes = self._pq.codebooks_.size * 4
        return {
            "index_type": "ivfpq",
            "metric": self.metric,
            "dim": self.dim,
            "ntotal": self._size,
            "is_trained": self.is_trained,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "m": self.m,
            "nbits": self._pq.nbits,
            "bytes_per_vector": self.m,
            "compression_ratio": round(self._pq.compression_ratio, 2),
            "memory_mb": round((code_bytes + centroid_bytes + codebook_bytes) / 1e6, 3),
            "codes_mb": round(code_bytes / 1e6, 3),
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
            "codes": self.codes.copy(),
            "ids": self._ids,
            "centroids": self.centroids_,
            "codebooks": self._pq.codebooks_,
            "list_offsets": offsets,
            "list_entries": flat,
            "dim": self.dim,
            "metric": self.metric,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "m": self.m,
            "nbits": self._pq.nbits,
            "kmeans_iter": self.kmeans_iter,
            "pq_iter": self._pq.n_iter,
            "pq_train_size": self.pq_train_size,
            "seed": self.seed,
            "block_size": self.block_size,
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> IVFPQIndex:
        index = cls(
            dim=int(state["dim"]),
            nlist=int(state["nlist"]),
            m=int(state["m"]),
            nbits=int(state["nbits"]),
            metric=str(state["metric"]),
            nprobe=int(state["nprobe"]),
            kmeans_iter=int(state["kmeans_iter"]),
            pq_iter=int(state["pq_iter"]),
            pq_train_size=int(state.get("pq_train_size", 100_000)),
            seed=int(state["seed"]),
            block_size=int(state["block_size"]),
        )
        index.centroids_ = np.asarray(state["centroids"], dtype=np.float32)
        index._centroid_sqnorms = squared_norms(index.centroids_)
        index._pq.codebooks_ = np.asarray(state["codebooks"], dtype=np.float32)
        codes = np.asarray(state["codes"], dtype=np.uint8)
        index._ensure_capacity(codes.shape[0])
        index._codes[: codes.shape[0]] = codes
        index._size = codes.shape[0]
        index._ids = np.asarray(state["ids"], dtype=np.int64)
        index._lists = InvertedLists.from_csr(state["list_offsets"], state["list_entries"])
        return index
