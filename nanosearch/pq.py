"""Product quantisation: store a vector as bytes instead of floats.

The idea
--------
A 128-d float32 vector costs 512 bytes. Split it into ``m`` contiguous
sub-vectors, run k-means with 256 centroids on each slice independently, and a
vector becomes ``m`` bytes -- one codebook index per slice. With ``m = 16``
that is 16 bytes instead of 512, a **32x** reduction, and the codebooks
together describe ``256^16 ~ 10^38`` distinct points, so the representable set
is enormous despite each slice being coarse.

Why it is fast as well as small (ADC)
-------------------------------------
The naive way to use a compressed index would be to decompress each candidate
and compute a real distance. Asymmetric Distance Computation avoids that. For a
query ``q``, precompute a table of distances from each of its sub-vectors to
each of the 256 centroids in that subspace: ``m * 256`` numbers. Then for any
stored vector, the approximate squared distance is a sum of ``m`` table
look-ups::

    d(q, x) ~ sum_j  table[j][ code_x[j] ]

Scoring a candidate costs ``m`` byte-indexed look-ups and ``m`` adds rather
than ``d`` multiply-adds, and -- the part that actually matters on modern
hardware -- 16 bytes per candidate instead of 512 means an entire inverted
list fits in cache.

"Asymmetric" refers to the query staying in full precision while only the
database side is quantised: strictly more accurate than quantising both.

A note on the batched matmul
----------------------------
The obvious implementation of ``fit``, ``encode`` and ``distance_tables`` is a
Python loop over the ``m`` subspaces, each doing its own small distance
computation. It is correct, and it was the first version here -- but profiling
showed it dominated query latency: ``m`` BLAS calls on ``(1, 8) x (8, 256)``
matrices spend all their time in call overhead, not arithmetic. Every such loop
below is instead expressed as one *batched* matmul over the subspace axis,
which hands BLAS a single job. See ``docs/DESIGN.md`` for the measured
before/after.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .kmeans import KMeans
from .metrics import FloatArray, as_float32, squared_l2

__all__ = ["ProductQuantizer"]


class ProductQuantizer:
    """Trains ``m`` independent 256-entry codebooks, one per sub-vector.

    Parameters
    ----------
    dim:
        Dimensionality of the vectors to compress.
    m:
        Number of sub-vectors. Must divide ``dim``. Larger ``m`` means finer
        quantisation (better recall) and more bytes per vector -- it is the
        main accuracy/size dial.
    nbits:
        Bits per sub-vector code. Fixed at 8: one code is then exactly one
        ``uint8``, which is what makes the look-up step a single cache-friendly
        gather.
    chunk_size:
        Rows encoded per pass. The intermediate distance array is
        ``(m, chunk, 256)``, so this bounds encode memory at roughly
        ``chunk * m * 1 KB`` -- without it, encoding a million vectors would
        ask for tens of gigabytes.
    """

    def __init__(
        self,
        dim: int,
        m: int = 8,
        nbits: int = 8,
        n_iter: int = 25,
        seed: int = 0,
        chunk_size: int = 4096,
    ):
        if dim % m != 0:
            raise ValueError(f"m={m} must divide dim={dim} (got remainder {dim % m})")
        if nbits != 8:
            raise ValueError("only nbits=8 is supported (one byte per sub-code)")
        self.dim = int(dim)
        self.m = int(m)
        self.nbits = int(nbits)
        self.ksub = 1 << self.nbits
        self.dsub = self.dim // self.m
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)

        self._codebooks: FloatArray = np.zeros((0, 0, 0), dtype=np.float32)
        self._codebooks_t: FloatArray = np.zeros((0, 0, 0), dtype=np.float32)
        self._codebook_sqnorms: FloatArray = np.zeros((0, 0), dtype=np.float32)

    # ------------------------------------------------------------- codebooks

    @property
    def codebooks_(self) -> FloatArray:
        """Learned codebooks, shape ``(m, ksub, dsub)``."""
        return self._codebooks

    @codebooks_.setter
    def codebooks_(self, value: Any) -> None:
        """Set codebooks and refresh the derived arrays used in the hot path.

        ``||centroid||^2`` and the transposed codebooks are both needed on
        every query and never change between queries, so they are computed
        once here rather than per search.
        """
        books = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        self._codebooks = books
        if books.size:
            if books.shape != (self.m, self.ksub, self.dsub):
                raise ValueError(
                    f"expected codebooks of shape {(self.m, self.ksub, self.dsub)}, "
                    f"got {books.shape}"
                )
            self._codebooks_t = np.ascontiguousarray(books.transpose(0, 2, 1))
            self._codebook_sqnorms = np.einsum("jkd,jkd->jk", books, books).astype(np.float32)
        else:
            self._codebooks_t = np.zeros((0, 0, 0), dtype=np.float32)
            self._codebook_sqnorms = np.zeros((0, 0), dtype=np.float32)

    # ------------------------------------------------------------ properties

    @property
    def is_trained(self) -> bool:
        return self._codebooks.size > 0

    @property
    def code_size(self) -> int:
        """Bytes used per stored vector."""
        return self.m

    @property
    def compression_ratio(self) -> float:
        """How much smaller than float32 storage."""
        return (self.dim * 4) / self.code_size

    # ------------------------------------------------------------------- fit

    def fit(self, vectors: Any) -> ProductQuantizer:
        """Learn the codebooks from a sample of vectors.

        Each subspace is clustered independently. That independence is the
        reason PQ scales: ``m`` cheap ``dsub``-dimensional clusterings instead
        of one hopeless ``dim``-dimensional one, and ``256^m`` effective
        centroids instead of 256. It also encodes the method's one real
        assumption -- that the slices carry roughly independent information.
        Slices of a learned embedding are not truly independent, which is what
        OPQ (a learned rotation applied before splitting) exists to fix.
        """
        x = self._check_dim(vectors)
        if x.shape[0] < self.ksub:
            import warnings

            warnings.warn(
                f"training PQ on {x.shape[0]} vectors for {self.ksub} centroids per subspace; "
                f"use at least {self.ksub} (ideally {39 * self.ksub}) for stable codebooks",
                UserWarning,
                stacklevel=2,
            )

        codebooks = np.zeros((self.m, self.ksub, self.dsub), dtype=np.float32)
        for j in range(self.m):
            sub = np.ascontiguousarray(x[:, j * self.dsub : (j + 1) * self.dsub])
            km = KMeans(n_clusters=self.ksub, n_iter=self.n_iter, seed=self.seed + j).fit(sub)
            codebooks[j] = km.centroids_
        self.codebooks_ = codebooks
        return self

    # -------------------------------------------------------- encode /decode

    def encode(self, vectors: Any) -> np.ndarray:
        """Compress vectors to ``(n, m)`` uint8 codes.

        For each subspace, pick the nearest codebook entry. ``||x||^2`` is
        dropped from the distance because it is constant across the centroids
        being compared and so cannot change the argmin -- one of those
        shortcuts that is obvious written down and easy to miss in code.
        """
        self._check_trained()
        x = self._check_dim(vectors)
        n = x.shape[0]
        codes = np.empty((n, self.m), dtype=np.uint8)

        for start in range(0, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            # (m, chunk, dsub) @ (m, dsub, ksub) -> (m, chunk, ksub), batched.
            sub = x[start:stop].reshape(stop - start, self.m, self.dsub).transpose(1, 0, 2)
            dist = np.matmul(np.ascontiguousarray(sub), self._codebooks_t)
            dist *= -2.0
            dist += self._codebook_sqnorms[:, None, :]
            codes[start:stop] = np.argmin(dist, axis=2).T.astype(np.uint8)

        return codes

    def decode(self, codes: np.ndarray) -> FloatArray:
        """Reconstruct approximate vectors from codes.

        Only needed for diagnostics and re-ranking experiments -- search itself
        never decodes, which is the whole point of ADC.
        """
        self._check_trained()
        codes = self._check_codes(codes)
        out = np.empty((codes.shape[0], self.dim), dtype=np.float32)
        for j in range(self.m):
            out[:, j * self.dsub : (j + 1) * self.dsub] = self._codebooks[j][codes[:, j]]
        return out

    # -------------------------------------------------------------- ADC path

    def distance_tables(self, queries: Any) -> FloatArray:
        """Per-query look-up tables, shape ``(n_queries, m, ksub)``.

        ``table[i, j, c]`` is the squared distance from query ``i``'s ``j``-th
        sub-vector to centroid ``c`` of subspace ``j``. Cost is
        ``ksub * dim`` flops per query, paid once and then amortised over every
        candidate that query scores -- which is why PQ wins on long candidate
        lists and loses on very short ones.
        """
        self._check_trained()
        q = self._check_dim(queries)
        nq = q.shape[0]

        sub = q.reshape(nq, self.m, self.dsub).transpose(1, 0, 2)
        sub = np.ascontiguousarray(sub)
        tables = np.matmul(sub, self._codebooks_t)  # (m, nq, ksub)
        tables *= -2.0
        tables += self._codebook_sqnorms[:, None, :]
        tables += np.einsum("jnd,jnd->jn", sub, sub)[:, :, None]
        np.maximum(tables, 0.0, out=tables)
        return np.ascontiguousarray(tables.transpose(1, 0, 2))

    def asymmetric_distances(
        self, queries: Any, codes: np.ndarray, tables: FloatArray | None = None
    ) -> FloatArray:
        """Approximate squared distances between queries and coded vectors.

        Shape ``(n_queries, n_codes)``. The loop runs over the ``m`` subspaces,
        not over the candidates: each iteration is one vectorised gather of
        shape ``(n_queries, n_codes)``, so interpreter overhead is O(m)
        regardless of how many candidates are scored.
        """
        self._check_trained()
        codes = self._check_codes(codes)
        if tables is None:
            tables = self.distance_tables(queries)
        out = np.zeros((tables.shape[0], codes.shape[0]), dtype=np.float32)
        for j in range(self.m):
            out += tables[:, j, :][:, codes[:, j]]
        return out

    def asymmetric_distances_grouped(
        self, tables: FloatArray, codes: np.ndarray, groups: np.ndarray
    ) -> FloatArray:
        """ADC where each candidate is scored against *its own* look-up table.

        IVF-PQ encodes residuals, so a candidate's table depends on which cell
        it lives in. The obvious implementation is a Python loop over cells,
        but then per-query cost scales with ``nprobe`` in the interpreter --
        which in practice wipes out the algorithmic win.

        Instead, ``groups[i]`` names the table to use for candidate ``i``, and
        the whole probed candidate set is scored in ``m`` vectorised gathers
        whatever ``nprobe`` was: O(m) interpreter work per query instead of
        O(nprobe * m).
        """
        self._check_trained()
        codes = self._check_codes(codes)
        groups = np.asarray(groups, dtype=np.int64)
        if codes.shape[0] != groups.shape[0]:
            raise ValueError("codes and groups must describe the same candidates")
        out = np.zeros(codes.shape[0], dtype=np.float32)
        for j in range(self.m):
            out += tables[groups, j, codes[:, j]]
        return out

    # ----------------------------------------------------------- diagnostics

    def reconstruction_error(self, vectors: Any) -> dict[str, float]:
        """Mean squared and relative error introduced by compression.

        Relative error is the fraction of each vector's energy lost, so 0.05
        means the stored approximation keeps 95% of it. Recall@k is the metric
        that matters downstream, but this one isolates the quantiser from the
        index.
        """
        x = self._check_dim(vectors)
        residual = x - self.decode(self.encode(x))
        mse = float(np.mean(np.einsum("ij,ij->i", residual, residual)))
        energy = float(np.mean(np.einsum("ij,ij->i", x, x)))
        return {
            "mse": mse,
            "relative_error": mse / max(energy, 1e-12),
            "bytes_per_vector": float(self.code_size),
            "compression_ratio": self.compression_ratio,
        }

    def reference_distance_tables(self, queries: Any) -> FloatArray:
        """Readable, slow version of :meth:`distance_tables`, for tests.

        Keeping the obvious implementation around and asserting the fast one
        agrees with it is cheap insurance: the batched-matmul version is a
        transpose-and-reshape puzzle, exactly the kind of code that is wrong in
        a way unit tests on *outputs* would never localise.
        """
        self._check_trained()
        q = self._check_dim(queries)
        tables = np.empty((q.shape[0], self.m, self.ksub), dtype=np.float32)
        for j in range(self.m):
            sub = q[:, j * self.dsub : (j + 1) * self.dsub]
            tables[:, j, :] = squared_l2(sub, self._codebooks[j])
        return tables

    # -------------------------------------------------------------- internals

    def _check_dim(self, vectors: Any) -> FloatArray:
        x = as_float32(vectors)
        if x.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim}-d vectors, got {x.shape[1]}-d")
        return x

    def _check_codes(self, codes: Any) -> np.ndarray:
        arr = np.asarray(codes, dtype=np.uint8)
        if arr.ndim != 2 or arr.shape[1] != self.m:
            raise ValueError(f"expected codes of shape (n, {self.m}), got {arr.shape}")
        return arr

    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("ProductQuantizer is not trained; call fit() first")
