"""Two-stage retrieval: does exact re-ranking recover what PQ gave up?

The whole justification for ``RefineIndex`` is one claim -- that a compressed
index plus a cheap exact re-rank of a few hundred candidates beats the
compressed index alone, without putting the vectors back in RAM. These tests
check both halves of that: recall goes up, and it goes up *monotonically* with
shortlist depth, converging on exact search.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanosearch import IVFPQIndex, RefineIndex, recall_at_k


@pytest.fixture(scope="module")
def two_stage(vectors: np.ndarray, tmp_path_factory: pytest.TempPathFactory) -> RefineIndex:
    base = IVFPQIndex(dim=vectors.shape[1], nlist=64, m=8, metric="cosine", seed=0)
    base.train(vectors)
    path = tmp_path_factory.mktemp("refine") / "vectors.f32"
    index = RefineIndex(base, path=path, factor=8)
    index.add(vectors)
    return index


class TestAccuracy:
    def test_beats_the_compressed_index_alone(
        self, two_stage: RefineIndex, vectors: np.ndarray, queries: np.ndarray, gold: np.ndarray
    ) -> None:
        alone = IVFPQIndex(dim=vectors.shape[1], nlist=64, m=8, metric="cosine", seed=0)
        alone.train(vectors)
        alone.add(vectors)

        without = recall_at_k(alone.search(queries, k=10, nprobe=16)[0], gold[:, :10])
        with_rerank = recall_at_k(
            two_stage.search(queries, k=10, factor=16, nprobe=16)[0], gold[:, :10]
        )
        assert with_rerank > without

    def test_recall_improves_with_shortlist_depth(
        self, two_stage: RefineIndex, queries: np.ndarray, gold: np.ndarray
    ) -> None:
        recalls = [
            recall_at_k(two_stage.search(queries, k=10, factor=f, nprobe=32)[0], gold[:, :10])
            for f in (1, 2, 4, 8, 16, 32)
        ]
        assert recalls == sorted(recalls), recalls

    def test_converges_on_exact_search(
        self, two_stage: RefineIndex, queries: np.ndarray, gold: np.ndarray
    ) -> None:
        # A deep shortlist over every cell leaves nothing for the
        # approximation to get wrong.
        ids, _ = two_stage.search(queries, k=10, factor=200, nprobe=64)
        assert recall_at_k(ids, gold[:, :10]) > 0.98

    def test_scores_are_exact_not_approximate(
        self, two_stage: RefineIndex, vectors: np.ndarray, queries: np.ndarray
    ) -> None:
        # Re-ranking reads real vectors, so the returned score must be the
        # true cosine similarity, not an ADC estimate.
        from nanosearch.metrics import l2_normalize

        ids, scores = two_stage.search(queries[:5], k=5, factor=32, nprobe=64)
        normalised = l2_normalize(vectors)
        for row in range(5):
            query = l2_normalize(queries[row : row + 1])[0]
            for doc_id, score in zip(ids[row], scores[row], strict=True):
                if doc_id < 0:
                    continue
                assert score == pytest.approx(float(query @ normalised[doc_id]), abs=1e-4)


class TestStorage:
    def test_vector_file_holds_the_raw_float32_rows(
        self, two_stage: RefineIndex, vectors: np.ndarray
    ) -> None:
        expected = vectors.shape[0] * vectors.shape[1] * 4
        assert two_stage.path.stat().st_size == expected

    def test_resident_memory_stays_compressed(
        self, two_stage: RefineIndex, vectors: np.ndarray
    ) -> None:
        stats = two_stage.stats()
        # In RAM: m bytes per vector. On disk: the full float32 vectors.
        assert stats["bytes_per_vector"] == two_stage.base.stats()["bytes_per_vector"]
        assert stats["disk_mb"] > stats["resident_memory_mb"]
        assert stats["index_type"] == "refine(ivfpq)"

    def test_rejects_invalid_factor(self, vectors: np.ndarray, tmp_path) -> None:
        base = IVFPQIndex(dim=vectors.shape[1], nlist=16, m=8)
        with pytest.raises(ValueError, match="factor must be"):
            RefineIndex(base, path=tmp_path / "v.f32", factor=0)

    def test_direct_save_is_refused_with_guidance(self, two_stage: RefineIndex, tmp_path) -> None:
        # Saving silently without the vector file would produce an index that
        # loads fine and then cannot re-rank.
        with pytest.raises(NotImplementedError, match="base index"):
            two_stage.save(tmp_path / "nope.npz")

    def test_empty_index_returns_padding(self, vectors: np.ndarray, tmp_path) -> None:
        base = IVFPQIndex(dim=vectors.shape[1], nlist=16, m=8, seed=0)
        base.train(vectors)
        index = RefineIndex(base, path=tmp_path / "empty.f32")
        ids, _ = index.search(vectors[:2], k=5)
        assert (ids == -1).all()
