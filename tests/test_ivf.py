"""IVF: does probing more cells really buy recall, and does probing all of them
reduce to exact search?

That last property is the one worth asserting hardest. ``nprobe = nlist`` means
every cell is visited, so the candidate set is the whole database and the
result must be *identical* to brute force. If it is not, the index is losing
vectors somewhere -- in the cell assignment, in the inverted lists, or in the
top-k merge -- and no amount of recall tuning would fix that.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanosearch import FlatIndex, IVFIndex, recall_at_k


@pytest.fixture(scope="module")
def built(vectors: np.ndarray) -> IVFIndex:
    index = IVFIndex(dim=vectors.shape[1], nlist=64, metric="cosine", seed=0)
    index.train(vectors)
    index.add(vectors)
    return index


class TestLifecycle:
    def test_is_untrained_until_trained(self, vectors: np.ndarray) -> None:
        index = IVFIndex(dim=vectors.shape[1], nlist=16)
        assert not index.is_trained
        index.train(vectors)
        assert index.is_trained

    def test_rejects_add_before_train(self, vectors: np.ndarray) -> None:
        index = IVFIndex(dim=vectors.shape[1], nlist=16)
        with pytest.raises(RuntimeError, match="must be trained"):
            index.add(vectors)

    def test_rejects_search_before_train(self, vectors: np.ndarray) -> None:
        index = IVFIndex(dim=vectors.shape[1], nlist=16)
        with pytest.raises(RuntimeError, match="must be trained"):
            index.search(vectors[:1], k=5)

    def test_rejects_training_set_smaller_than_nlist(self, vectors: np.ndarray) -> None:
        index = IVFIndex(dim=vectors.shape[1], nlist=500)
        with pytest.raises(ValueError, match="at least nlist"):
            index.train(vectors[:100])

    def test_rejects_non_positive_nprobe(self, vectors: np.ndarray) -> None:
        with pytest.raises(ValueError, match="nprobe must be positive"):
            IVFIndex(dim=vectors.shape[1], nprobe=0)


class TestRecall:
    def test_probing_every_cell_equals_exact_search(
        self, built: IVFIndex, vectors: np.ndarray, queries: np.ndarray, gold: np.ndarray
    ) -> None:
        ids, _ = built.search(queries, k=10, nprobe=built.nlist)
        assert recall_at_k(ids, gold[:, :10]) == pytest.approx(1.0)

    def test_recall_increases_with_nprobe(
        self, built: IVFIndex, queries: np.ndarray, gold: np.ndarray
    ) -> None:
        recalls = [
            recall_at_k(built.search(queries, k=10, nprobe=nprobe)[0], gold[:, :10])
            for nprobe in (1, 2, 4, 8, 16, 32, 64)
        ]
        assert recalls == sorted(recalls), recalls
        assert recalls[0] < recalls[-1]

    def test_single_cell_probe_is_still_useful(
        self, built: IVFIndex, queries: np.ndarray, gold: np.ndarray
    ) -> None:
        # Probing one cell out of 64 should still find most neighbours on
        # clustered data -- that is the premise of the whole approach.
        recall = recall_at_k(built.search(queries, k=10, nprobe=1)[0], gold[:, :10])
        assert recall > 0.5

    def test_nprobe_is_clamped_to_nlist(self, built: IVFIndex, queries: np.ndarray) -> None:
        huge = built.search(queries[:5], k=10, nprobe=10_000)[0]
        exact = built.search(queries[:5], k=10, nprobe=built.nlist)[0]
        np.testing.assert_array_equal(huge, exact)

    def test_scores_match_exact_scores_for_returned_ids(
        self, built: IVFIndex, vectors: np.ndarray, queries: np.ndarray
    ) -> None:
        # Approximation may change *which* vectors come back, never the score
        # computed for the ones that do: IVF stores full-precision vectors.
        flat = FlatIndex(dim=vectors.shape[1], metric="cosine")
        flat.add(vectors)
        ids, scores = built.search(queries[:5], k=5, nprobe=built.nlist)
        flat_ids, flat_scores = flat.search(queries[:5], k=5)
        np.testing.assert_array_equal(ids, flat_ids)
        np.testing.assert_allclose(scores, flat_scores, atol=1e-5)


class TestInvertedLists:
    def test_every_vector_lands_in_exactly_one_list(
        self, built: IVFIndex, vectors: np.ndarray
    ) -> None:
        sizes = built._lists.sizes()
        assert int(sizes.sum()) == vectors.shape[0]
        entries = np.concatenate(
            [built._lists[i] for i in range(built.nlist) if built._lists[i].size]
        )
        assert len(set(entries.tolist())) == vectors.shape[0]

    def test_imbalance_factor_is_reported_and_sane(self, built: IVFIndex) -> None:
        stats = built.stats()
        assert stats["imbalance_factor"] >= 1.0
        assert stats["list_size_max"] >= stats["list_size_mean"]

    def test_stats_describe_an_approximate_full_precision_index(self, built: IVFIndex) -> None:
        stats = built.stats()
        assert stats["index_type"] == "ivf"
        assert stats["exact"] is False
        assert stats["bytes_per_vector"] == built.dim * 4


class TestCloneTrained:
    def test_clone_shares_centroids_but_holds_no_vectors(self, built: IVFIndex) -> None:
        clone = built.clone_trained()
        assert clone.is_trained
        assert clone.ntotal == 0
        np.testing.assert_array_equal(clone.centroids_, built.centroids_)

    def test_clone_can_index_a_disjoint_shard(
        self, built: IVFIndex, vectors: np.ndarray, queries: np.ndarray
    ) -> None:
        # Sharding: two clones index half the data each, and because they share
        # the quantiser their scores are directly comparable, so merging their
        # results reproduces the single-index answer.
        left, right = built.clone_trained(), built.clone_trained()
        half = vectors.shape[0] // 2
        left.add(vectors[:half], ids=np.arange(half))
        right.add(vectors[half:], ids=np.arange(half, vectors.shape[0]))

        query = queries[:1]
        merged_ids = np.concatenate(
            [left.search(query, k=10, nprobe=64)[0][0], right.search(query, k=10, nprobe=64)[0][0]]
        )
        merged_scores = np.concatenate(
            [
                left.search(query, k=10, nprobe=64)[1][0],
                right.search(query, k=10, nprobe=64)[1][0],
            ]
        )
        top = merged_ids[np.argsort(-merged_scores)][:10]
        single = built.search(query, k=10, nprobe=64)[0][0]
        assert set(top.tolist()) == set(single.tolist())
