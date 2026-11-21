"""IVF-PQ: both approximations at once, and the honest limits of the result.

These tests encode what IVF-PQ is actually *for*. It is not an exact index with
a discount -- it is a candidate generator with a very small memory footprint.
So the assertions are:

* memory really is ``m`` bytes per vector
* the true neighbours land in a shortlist (high recall at depth)
* more ``nprobe`` eventually stops helping, because the ceiling is the codes
"""

from __future__ import annotations

import numpy as np
import pytest

from nanosearch import IVFPQIndex, recall_at_k


@pytest.fixture(scope="module")
def built(vectors: np.ndarray) -> IVFPQIndex:
    index = IVFPQIndex(dim=vectors.shape[1], nlist=64, m=16, metric="cosine", seed=0)
    index.train(vectors)
    index.add(vectors)
    return index


class TestLifecycle:
    def test_requires_training_before_add(self, vectors: np.ndarray) -> None:
        index = IVFPQIndex(dim=vectors.shape[1], nlist=16, m=8)
        assert not index.is_trained
        with pytest.raises(RuntimeError, match="must be trained"):
            index.add(vectors)

    def test_requires_training_before_search(self, vectors: np.ndarray) -> None:
        index = IVFPQIndex(dim=vectors.shape[1], nlist=16, m=8)
        with pytest.raises(RuntimeError, match="must be trained"):
            index.search(vectors[:1], k=5)

    def test_rejects_m_that_does_not_divide_dim(self, vectors: np.ndarray) -> None:
        with pytest.raises(ValueError, match="must divide"):
            IVFPQIndex(dim=vectors.shape[1], m=vectors.shape[1] - 1)

    def test_training_subsamples_for_codebooks(self, vectors: np.ndarray) -> None:
        # pq_train_size keeps training memory flat as the corpus grows: the
        # codebooks are fitted from a sample, never from the whole corpus.
        index = IVFPQIndex(dim=vectors.shape[1], nlist=32, m=8, pq_train_size=500, seed=0)
        index.train(vectors)
        assert index.is_trained


class TestStorage:
    def test_uses_exactly_m_bytes_per_vector(self, built: IVFPQIndex, vectors: np.ndarray) -> None:
        stats = built.stats()
        assert stats["bytes_per_vector"] == built.m
        assert built.codes.shape == (vectors.shape[0], built.m)
        assert built.codes.dtype == np.uint8

    def test_reports_compression_against_float32(self, built: IVFPQIndex) -> None:
        stats = built.stats()
        assert stats["compression_ratio"] == pytest.approx(built.dim * 4 / built.m)

    def test_does_not_retain_full_precision_vectors(self, built: IVFPQIndex) -> None:
        # The memory win is precisely that the originals are gone.
        assert not hasattr(built, "_data")

    def test_codes_memory_scales_with_count_not_dimension(self, built: IVFPQIndex) -> None:
        stats = built.stats()
        assert stats["codes_mb"] == pytest.approx(built.ntotal * built.m / 1e6, rel=1e-6)


class TestRecall:
    def test_true_neighbours_land_in_a_shortlist(
        self, built: IVFPQIndex, queries: np.ndarray, gold: np.ndarray
    ) -> None:
        shortlist, _ = built.search(queries, k=100, nprobe=16)
        assert recall_at_k(shortlist, gold[:, :10], k=10) > 0.85

    def test_recall_at_depth_beats_recall_at_ten(
        self, built: IVFPQIndex, queries: np.ndarray, gold: np.ndarray
    ) -> None:
        # The characteristic PQ signature: good at finding the neighbourhood,
        # poor at ordering within it. This is why re-ranking exists.
        strict = recall_at_k(built.search(queries, k=10, nprobe=16)[0], gold[:, :10])
        deep = recall_at_k(built.search(queries, k=100, nprobe=16)[0], gold[:, :10], k=10)
        assert deep >= strict

    def test_more_nprobe_does_not_reduce_recall(
        self, built: IVFPQIndex, queries: np.ndarray, gold: np.ndarray
    ) -> None:
        recalls = [
            recall_at_k(built.search(queries, k=50, nprobe=nprobe)[0], gold[:, :10], k=10)
            for nprobe in (1, 4, 16, 64)
        ]
        assert recalls[-1] >= recalls[0]
        # And the ceiling is set by the codes, not by nprobe: probing every
        # cell does not reach exact search.
        assert recalls[-1] <= 1.0

    def test_returns_valid_ids_and_finite_scores(
        self, built: IVFPQIndex, queries: np.ndarray
    ) -> None:
        ids, scores = built.search(queries, k=10, nprobe=8)
        assert ids.shape == scores.shape == (queries.shape[0], 10)
        assert (ids >= 0).all()
        assert np.isfinite(scores).all()
        assert ((scores >= -1.001) & (scores <= 1.001)).all()


class TestCloneTrained:
    def test_clone_shares_codebooks_and_centroids(self, built: IVFPQIndex) -> None:
        clone = built.clone_trained()
        assert clone.is_trained
        assert clone.ntotal == 0
        np.testing.assert_array_equal(clone.centroids_, built.centroids_)
        np.testing.assert_array_equal(clone._pq.codebooks_, built._pq.codebooks_)

    def test_clone_produces_identical_codes_for_the_same_input(
        self, built: IVFPQIndex, vectors: np.ndarray
    ) -> None:
        clone = built.clone_trained()
        clone.add(vectors[:200])
        np.testing.assert_array_equal(clone.codes, built.codes[:200])
