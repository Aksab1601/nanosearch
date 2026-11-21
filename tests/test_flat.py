"""Exact search. If this is wrong, every recall number in the project is wrong."""

from __future__ import annotations

import numpy as np
import pytest

from nanosearch import FlatIndex
from nanosearch.metrics import l2_normalize, squared_l2


class TestCorrectness:
    def test_finds_each_vector_as_its_own_nearest_neighbour(self, vectors: np.ndarray) -> None:
        index = FlatIndex(dim=vectors.shape[1])
        index.add(vectors)
        ids, scores = index.search(vectors[:25], k=1)
        np.testing.assert_array_equal(ids.ravel(), np.arange(25))
        np.testing.assert_allclose(scores.ravel(), 1.0, atol=1e-4)

    def test_matches_brute_force_numpy_ranking(self, vectors: np.ndarray) -> None:
        index = FlatIndex(dim=vectors.shape[1], metric="cosine")
        index.add(vectors)
        queries = vectors[100:105]

        ids, _ = index.search(queries, k=10)
        similarity = l2_normalize(queries) @ l2_normalize(vectors).T
        expected = np.argsort(-similarity, axis=1)[:, :10]
        np.testing.assert_array_equal(ids, expected)

    def test_l2_metric_ranks_by_distance(self, vectors: np.ndarray) -> None:
        index = FlatIndex(dim=vectors.shape[1], metric="l2")
        index.add(vectors)
        ids, scores = index.search(vectors[7:8], k=5)
        expected = np.argsort(squared_l2(vectors[7:8], vectors)[0])[:5]
        np.testing.assert_array_equal(ids.ravel(), expected)
        # Scores are true distances, so ascending and zero for the self-match.
        assert scores[0, 0] == pytest.approx(0.0, abs=1e-3)
        assert (np.diff(scores.ravel()) >= 0).all()

    def test_block_size_does_not_change_results(self, vectors: np.ndarray) -> None:
        # Blocking exists for memory, so it must be invisible in the output:
        # the merge of per-block top-k has to reproduce the global top-k.
        small = FlatIndex(dim=vectors.shape[1], block_size=64)
        small.add(vectors)
        big = FlatIndex(dim=vectors.shape[1], block_size=10_000_000)
        big.add(vectors)
        np.testing.assert_array_equal(
            small.search(vectors[:12], k=20)[0], big.search(vectors[:12], k=20)[0]
        )


class TestGrowth:
    def test_incremental_adds_match_one_bulk_add(self, vectors: np.ndarray) -> None:
        bulk = FlatIndex(dim=vectors.shape[1])
        bulk.add(vectors[:500])

        incremental = FlatIndex(dim=vectors.shape[1])
        for start in range(0, 500, 37):  # forces several capacity doublings
            incremental.add(vectors[start : min(start + 37, 500)])

        assert incremental.ntotal == bulk.ntotal == 500
        np.testing.assert_array_equal(
            incremental.search(vectors[:5], k=10)[0], bulk.search(vectors[:5], k=10)[0]
        )

    def test_assigns_sequential_ids_by_default(self, vectors: np.ndarray) -> None:
        index = FlatIndex(dim=vectors.shape[1])
        np.testing.assert_array_equal(index.add(vectors[:3]), [0, 1, 2])
        np.testing.assert_array_equal(index.add(vectors[3:5]), [3, 4])

    def test_honours_caller_supplied_ids(self, vectors: np.ndarray) -> None:
        index = FlatIndex(dim=vectors.shape[1])
        index.add(vectors[:3], ids=[100, 200, 300])
        ids, _ = index.search(vectors[1:2], k=1)
        assert ids[0, 0] == 200

    def test_rejects_mismatched_id_count(self, vectors: np.ndarray) -> None:
        index = FlatIndex(dim=vectors.shape[1])
        with pytest.raises(ValueError, match="ids for"):
            index.add(vectors[:3], ids=[1, 2])


class TestEdgeCases:
    def test_empty_index_returns_padding(self) -> None:
        ids, scores = FlatIndex(dim=4).search(np.zeros((2, 4), dtype=np.float32), k=3)
        assert ids.shape == (2, 3)
        assert (ids == -1).all()
        assert np.isneginf(scores).all()

    def test_k_larger_than_index_pads_the_tail(self, vectors: np.ndarray) -> None:
        index = FlatIndex(dim=vectors.shape[1])
        index.add(vectors[:4])
        ids, scores = index.search(vectors[:1], k=10)
        assert ids.shape == (1, 10)
        assert (ids[0, :4] >= 0).all()
        assert (ids[0, 4:] == -1).all()
        assert np.isneginf(scores[0, 4:]).all()

    def test_rejects_wrong_dimensionality(self, vectors: np.ndarray) -> None:
        index = FlatIndex(dim=vectors.shape[1])
        with pytest.raises(ValueError, match="index is"):
            index.add(np.zeros((2, vectors.shape[1] + 1), dtype=np.float32))

    def test_rejects_non_finite_vectors(self) -> None:
        index = FlatIndex(dim=3)
        with pytest.raises(ValueError, match="NaN or inf"):
            index.add(np.array([[1.0, np.nan, 2.0]], dtype=np.float32))

    def test_rejects_unknown_metric(self) -> None:
        with pytest.raises(ValueError, match="metric must be"):
            FlatIndex(dim=4, metric="manhattan")

    def test_rejects_non_positive_dim(self) -> None:
        with pytest.raises(ValueError, match="dim must be positive"):
            FlatIndex(dim=0)


class TestStats:
    def test_reports_exact_and_full_precision_storage(self, vectors: np.ndarray) -> None:
        index = FlatIndex(dim=vectors.shape[1])
        index.add(vectors)
        stats = index.stats()
        assert stats["exact"] is True
        assert stats["bytes_per_vector"] == vectors.shape[1] * 4
        assert stats["ntotal"] == vectors.shape[0]
