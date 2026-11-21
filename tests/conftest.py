"""Shared fixtures.

Session-scoped so the clustering and SVD work happens once for the whole run
rather than once per test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanosearch import FlatIndex
from nanosearch.engine import SearchEngine
from nanosearch.store import DocumentStore

CORPUS = Path(__file__).resolve().parent.parent / "nanosearch" / "data" / "corpus.jsonl"

DIM = 32
N_VECTORS = 4000
N_QUERIES = 60


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def vectors() -> np.ndarray:
    """Clustered vectors: the structure an IVF index is supposed to exploit."""
    generator = np.random.default_rng(7)
    centers = generator.normal(size=(40, DIM)).astype(np.float32) * 2.0
    assignments = generator.integers(0, 40, size=N_VECTORS)
    noise = generator.normal(scale=0.6, size=(N_VECTORS, DIM)).astype(np.float32)
    return np.ascontiguousarray(centers[assignments] + noise, dtype=np.float32)


@pytest.fixture(scope="session")
def queries() -> np.ndarray:
    generator = np.random.default_rng(8)
    centers = np.random.default_rng(7).normal(size=(40, DIM)).astype(np.float32) * 2.0
    assignments = generator.integers(0, 40, size=N_QUERIES)
    noise = generator.normal(scale=0.6, size=(N_QUERIES, DIM)).astype(np.float32)
    return np.ascontiguousarray(centers[assignments] + noise, dtype=np.float32)


@pytest.fixture(scope="session")
def gold(vectors: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Exact top-100 ids per query: the ground truth for every recall check."""
    index = FlatIndex(dim=DIM, metric="cosine")
    index.add(vectors)
    return index.search(queries, k=100)[0]


@pytest.fixture(scope="session")
def corpus() -> tuple[list[str], list[dict]]:
    return DocumentStore.from_jsonl(CORPUS)


@pytest.fixture(scope="session")
def engine(corpus: tuple[list[str], list[dict]]) -> SearchEngine:
    texts, metadatas = corpus
    return SearchEngine.build(texts, metadatas=metadatas)
