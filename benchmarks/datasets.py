"""Benchmark data: synthetic vectors shaped like real embeddings, or your own.

Why not uniform random vectors
------------------------------
Benchmarking an ANN index on ``rng.normal(size=(n, d))`` flatters it twice
over. Isotropic Gaussian points in high dimensions are all roughly equidistant,
so there is no neighbourhood structure to find and no clustering for IVF to
exploit -- and ironically that makes recall curves look *better* than reality
once any clustering is injected, because well-separated blobs are trivially
partitioned.

Real sentence or image embeddings have two properties this generator
reproduces:

1. **Strong anisotropy.** Variance is concentrated in a few directions; the
   covariance spectrum decays roughly as a power law. Most of an embedding's
   effective dimensionality is far below its nominal one.
2. **Overlapping clusters.** Topics form clumps, but the clumps bleed into one
   another -- so some true neighbours of a query genuinely do live in a
   different cell, which is precisely the error mode IVF has.

Vectors are L2-normalised, like the output of every modern embedding model.

Honest limitations
------------------
These are synthetic vectors. Absolute recall at a given ``nprobe`` depends on
the distribution, so the numbers here describe *this* distribution, measured
against exact search on the same data. ``--data vectors.npy`` runs the same
harness on a real embedding set; see the README for how.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_embedding_like(
    n: int,
    dim: int,
    n_clusters: int = 64,
    spectrum_decay: float = 1.0,
    cluster_scale: float = 0.55,
    seed: int = 0,
) -> np.ndarray:
    """Generate ``n`` L2-normalised ``dim``-d vectors with embedding-like geometry.

    Parameters
    ----------
    n_clusters:
        Number of topical clumps.
    spectrum_decay:
        Exponent of the covariance spectrum: eigenvalue ``i`` scales as
        ``(i + 1) ** -spectrum_decay``. 0 is isotropic; 1.0 is close to what
        sentence embeddings show in practice.
    cluster_scale:
        Within-cluster spread relative to between-cluster spread. Larger means
        more overlap and a genuinely harder search problem.
    """
    rng = np.random.default_rng(seed)

    # Anisotropic covariance: a random rotation applied to a power-law spectrum.
    spectrum = (np.arange(1, dim + 1, dtype=np.float64) ** -spectrum_decay).astype(np.float32)
    rotation, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    transform = (rotation * spectrum[None, :]).astype(np.float32)

    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32) @ transform
    out = np.empty((n, dim), dtype=np.float32)

    # Generated in chunks: rng.normal returns float64, so a one-shot
    # (1e6, 128) draw costs a gigabyte before the float32 cast even starts.
    chunk = 65_536
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rows = stop - start
        noise = rng.normal(size=(rows, dim)).astype(np.float32) @ transform
        block = centers[rng.integers(0, n_clusters, size=rows)] + cluster_scale * noise
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        out[start:stop] = block / np.maximum(norms, 1e-12)

    return out


def load_or_make(
    path: str | Path | None,
    n: int,
    dim: int,
    seed: int = 0,
    **kwargs: float | int,
) -> tuple[np.ndarray, str]:
    """Load vectors from a ``.npy`` file, or synthesise them.

    Returns the vectors and a label describing their provenance, so the
    benchmark output always records which data produced the numbers.
    """
    if path is not None:
        vectors = np.load(Path(path)).astype(np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D array of vectors, got shape {vectors.shape}")
        return np.ascontiguousarray(vectors), f"real:{Path(path).name}"
    vectors = make_embedding_like(n=n, dim=dim, seed=seed, **kwargs)  # type: ignore[arg-type]
    return vectors, f"synthetic(n={n}, dim={dim}, seed={seed})"


def split_queries(
    vectors: np.ndarray, n_queries: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out ``n_queries`` rows as queries and return ``(database, queries)``.

    Queries are *removed* from the database. Searching for vectors that are
    themselves indexed is the most common way ANN benchmarks accidentally
    overstate recall: every query then has a perfect match at distance zero,
    which any index finds, and the hard part of the ranking disappears.
    """
    rng = np.random.default_rng(seed)
    if n_queries >= vectors.shape[0]:
        raise ValueError("n_queries must be smaller than the number of vectors")
    picked = rng.choice(vectors.shape[0], size=n_queries, replace=False)
    mask = np.ones(vectors.shape[0], dtype=bool)
    mask[picked] = False
    return np.ascontiguousarray(vectors[mask]), np.ascontiguousarray(vectors[picked])
