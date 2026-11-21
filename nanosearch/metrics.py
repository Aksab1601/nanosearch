"""Distance primitives and top-k selection.

Every index in this project (flat, IVF, IVF-PQ) ultimately does two things:

1. compute squared Euclidean distances between queries and some set of vectors
2. select the k smallest of those distances

Keeping both primitives in one module means the indexes differ only in *which*
candidates they score, never in *how* scoring works. That is the single most
useful structural decision in the codebase: it makes the recall/latency
trade-off the only moving part.

A note on cosine similarity
---------------------------
For L2-normalised vectors ``q`` and ``x``::

    ||q - x||^2 = ||q||^2 - 2 q.x + ||x||^2 = 2 - 2 cos(q, x)

Squared L2 distance is therefore a strictly decreasing function of cosine
similarity, so ranking by one is identical to ranking by the other. nanosearch
exploits this: when ``metric="cosine"`` it L2-normalises vectors on the way in
and then uses the same squared-L2 machinery everywhere, converting back to a
similarity only when results are returned. One code path, two metrics.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

__all__ = [
    "FloatArray",
    "IntArray",
    "as_float32",
    "cosine_from_sqdist",
    "l2_normalize",
    "squared_l2",
    "squared_norms",
    "topk_smallest",
]


def as_float32(x: npt.ArrayLike) -> FloatArray:
    """Return ``x`` as a contiguous 2-D float32 array.

    float32 halves memory versus float64 and is what every production vector
    store uses; embedding models emit far less than 7 significant digits of
    real signal, so the precision is not the bottleneck.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def squared_norms(x: FloatArray) -> FloatArray:
    """Row-wise squared L2 norms, i.e. ``(x * x).sum(axis=1)``.

    ``einsum`` is used instead of ``(x ** 2).sum(1)`` because it avoids
    materialising a second array the size of ``x``.
    """
    return np.einsum("ij,ij->i", x, x).astype(np.float32, copy=False)


def l2_normalize(x: npt.ArrayLike, eps: float = 1e-12) -> FloatArray:
    """Scale each row to unit L2 norm. Zero rows are left as zeros."""
    arr = as_float32(x).copy()
    norms = np.sqrt(squared_norms(arr))
    np.divide(arr, np.maximum(norms, eps)[:, None], out=arr)
    return arr


def squared_l2(
    queries: npt.ArrayLike,
    vectors: npt.ArrayLike,
    vector_sqnorms: FloatArray | None = None,
) -> FloatArray:
    """Pairwise squared L2 distances, shape ``(n_queries, n_vectors)``.

    Uses the expansion ``||q - x||^2 = ||q||^2 - 2 q.x + ||x||^2`` so the work
    is dominated by one matrix multiply, which NumPy hands to BLAS. The naive
    ``((q[:, None] - x[None]) ** 2).sum(-1)`` is mathematically identical but
    allocates an ``(nq, n, d)`` intermediate and runs orders of magnitude
    slower.

    ``vector_sqnorms`` lets callers pass in precomputed ``||x||^2`` values.
    Indexes keep those cached because the database does not change between
    queries -- only the queries do.
    """
    q = as_float32(queries)
    x = as_float32(vectors)
    if q.shape[1] != x.shape[1]:
        raise ValueError(f"dimension mismatch: queries are {q.shape[1]}-d, vectors {x.shape[1]}-d")

    x_sq = squared_norms(x) if vector_sqnorms is None else vector_sqnorms
    if x_sq.shape[0] != x.shape[0]:
        raise ValueError("vector_sqnorms length does not match the number of vectors")

    dist = q @ x.T
    dist *= -2.0
    dist += x_sq[None, :]
    dist += squared_norms(q)[:, None]
    # The expansion can produce small negatives for near-identical vectors
    # through floating point cancellation; distances are never negative.
    np.maximum(dist, 0.0, out=dist)
    return dist


def topk_smallest(values: npt.ArrayLike, k: int) -> tuple[IntArray, FloatArray]:
    """Indices and values of the ``k`` smallest entries along the last axis, sorted.

    ``np.argpartition`` runs in O(n) (quickselect) and only the k selected
    entries are then sorted, for O(n + k log k) total. A full ``np.argsort`` is
    O(n log n) and does strictly more work than the question requires -- the
    same selection-vs-sorting distinction that shows up in "find the k largest
    elements" interview problems.
    """
    vals = np.asarray(values, dtype=np.float32)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n = vals.shape[-1]
    k = min(k, n)
    if k == n:
        idx = np.argsort(vals, axis=-1, kind="stable")
    else:
        part = np.argpartition(vals, k - 1, axis=-1)[..., :k]
        part_vals = np.take_along_axis(vals, part, axis=-1)
        order = np.argsort(part_vals, axis=-1, kind="stable")
        idx = np.take_along_axis(part, order, axis=-1)
    idx = idx[..., :k].astype(np.int64, copy=False)
    return idx, np.take_along_axis(vals, idx, axis=-1)


def cosine_from_sqdist(sqdist: FloatArray) -> FloatArray:
    """Recover cosine similarity from squared L2 distance of unit vectors."""
    return (1.0 - 0.5 * np.asarray(sqdist, dtype=np.float32)).astype(np.float32, copy=False)
