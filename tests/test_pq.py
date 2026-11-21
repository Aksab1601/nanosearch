"""Product quantisation: compression, and the ADC distance approximation.

The important tests here are not "does it run" but:

* the fast batched-matmul table builder agrees with the obvious loop
* ADC distances actually approximate true distances (if they did not, the
  index would be fast and meaningless)
* error shrinks as ``m`` grows, which is the accuracy/size dial working
"""

from __future__ import annotations

import numpy as np
import pytest

from nanosearch.metrics import squared_l2
from nanosearch.pq import ProductQuantizer


@pytest.fixture(scope="module")
def trained(vectors: np.ndarray) -> ProductQuantizer:
    return ProductQuantizer(dim=vectors.shape[1], m=8, seed=0).fit(vectors)


class TestConfiguration:
    def test_rejects_m_that_does_not_divide_dim(self) -> None:
        with pytest.raises(ValueError, match="must divide"):
            ProductQuantizer(dim=10, m=3)

    def test_rejects_nbits_other_than_eight(self) -> None:
        with pytest.raises(ValueError, match="nbits=8"):
            ProductQuantizer(dim=8, m=2, nbits=4)

    def test_reports_code_size_and_compression(self) -> None:
        pq = ProductQuantizer(dim=128, m=16)
        assert pq.code_size == 16
        assert pq.compression_ratio == pytest.approx(32.0)
        assert pq.dsub == 8

    def test_rejects_use_before_training(self, vectors: np.ndarray) -> None:
        pq = ProductQuantizer(dim=vectors.shape[1], m=4)
        assert not pq.is_trained
        with pytest.raises(RuntimeError, match="not trained"):
            pq.encode(vectors[:2])

    def test_warns_when_training_data_is_too_small(self) -> None:
        # 256 centroids per subspace cannot be fitted from 20 points; silently
        # returning garbage codebooks would be worse than complaining.
        with pytest.warns(UserWarning, match="centroids per subspace"):
            ProductQuantizer(dim=8, m=2, n_iter=1).fit(
                np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
            )


class TestEncoding:
    def test_codes_have_the_right_shape_and_dtype(
        self, trained: ProductQuantizer, vectors: np.ndarray
    ) -> None:
        codes = trained.encode(vectors[:50])
        assert codes.shape == (50, trained.m)
        assert codes.dtype == np.uint8

    def test_encoding_picks_the_nearest_centroid_per_subspace(
        self, trained: ProductQuantizer, vectors: np.ndarray
    ) -> None:
        sample = vectors[:20]
        codes = trained.encode(sample)
        for j in range(trained.m):
            sub = sample[:, j * trained.dsub : (j + 1) * trained.dsub]
            expected = np.argmin(squared_l2(sub, trained.codebooks_[j]), axis=1)
            np.testing.assert_array_equal(codes[:, j], expected)

    def test_chunking_does_not_change_codes(
        self, trained: ProductQuantizer, vectors: np.ndarray
    ) -> None:
        trained.chunk_size = 7
        chunked = trained.encode(vectors[:100])
        trained.chunk_size = 100_000
        np.testing.assert_array_equal(chunked, trained.encode(vectors[:100]))

    def test_decode_returns_the_chosen_centroids(
        self, trained: ProductQuantizer, vectors: np.ndarray
    ) -> None:
        codes = trained.encode(vectors[:10])
        decoded = trained.decode(codes)
        assert decoded.shape == (10, trained.dim)
        for j in range(trained.m):
            span = slice(j * trained.dsub, (j + 1) * trained.dsub)
            np.testing.assert_allclose(decoded[:, span], trained.codebooks_[j][codes[:, j]], atol=0)

    def test_rejects_codes_of_the_wrong_width(self, trained: ProductQuantizer) -> None:
        with pytest.raises(ValueError, match="expected codes of shape"):
            trained.decode(np.zeros((3, trained.m + 1), dtype=np.uint8))


class TestAccuracy:
    def test_reconstruction_error_improves_with_more_subspaces(self, vectors: np.ndarray) -> None:
        errors = [
            ProductQuantizer(dim=vectors.shape[1], m=m, seed=0)
            .fit(vectors)
            .reconstruction_error(vectors[:500])["relative_error"]
            for m in (2, 4, 8, 16)
        ]
        assert errors == sorted(errors, reverse=True), errors

    def test_adc_distances_approximate_true_distances(
        self, trained: ProductQuantizer, vectors: np.ndarray
    ) -> None:
        # The claim the whole index rests on. Not "equal" -- approximate,
        # within the quantiser's own measured error.
        queries, database = vectors[:8], vectors[1000:1300]
        codes = trained.encode(database)
        approximate = trained.asymmetric_distances(queries, codes)
        exact = squared_l2(queries, database)

        relative = np.abs(approximate - exact) / np.maximum(exact, 1e-6)
        assert float(np.median(relative)) < 0.25

    def test_adc_preserves_the_ranking_well_enough_to_shortlist(
        self, trained: ProductQuantizer, vectors: np.ndarray
    ) -> None:
        # PQ is a candidate generator: the true nearest neighbours must land
        # in a modest shortlist even if their exact order is scrambled.
        queries, database = vectors[:20], vectors[2000:2600]
        codes = trained.encode(database)
        approximate = trained.asymmetric_distances(queries, codes)
        exact = squared_l2(queries, database)

        found = 0
        for row in range(queries.shape[0]):
            true_top5 = set(np.argsort(exact[row])[:5].tolist())
            shortlist = set(np.argsort(approximate[row])[:50].tolist())
            found += len(true_top5 & shortlist)
        assert found / (20 * 5) > 0.8


class TestDistanceTables:
    def test_batched_tables_match_the_reference_loop(
        self, trained: ProductQuantizer, vectors: np.ndarray
    ) -> None:
        # The fast path is a transpose-and-reshape puzzle; the readable loop is
        # kept precisely so this equivalence can be asserted.
        queries = vectors[:17]
        np.testing.assert_allclose(
            trained.distance_tables(queries),
            trained.reference_distance_tables(queries),
            atol=1e-4,
        )

    def test_table_shape(self, trained: ProductQuantizer, vectors: np.ndarray) -> None:
        tables = trained.distance_tables(vectors[:4])
        assert tables.shape == (4, trained.m, trained.ksub)

    def test_tables_are_non_negative(self, trained: ProductQuantizer, vectors: np.ndarray) -> None:
        assert (trained.distance_tables(vectors[:30]) >= 0).all()

    def test_grouped_adc_matches_per_group_scoring(
        self, trained: ProductQuantizer, vectors: np.ndarray
    ) -> None:
        # The grouped kernel is what keeps IVF-PQ's interpreter cost O(m)
        # instead of O(nprobe * m); it must agree with the naive version.
        database = vectors[500:560]
        codes = trained.encode(database)
        residual_queries = vectors[:3]
        tables = trained.distance_tables(residual_queries)

        groups = np.repeat(np.arange(3), 20)
        grouped = trained.asymmetric_distances_grouped(tables, codes, groups)

        expected = np.concatenate(
            [
                trained.asymmetric_distances(
                    residual_queries[g : g + 1], codes[g * 20 : (g + 1) * 20]
                )[0]
                for g in range(3)
            ]
        )
        np.testing.assert_allclose(grouped, expected, atol=1e-5)

    def test_grouped_adc_rejects_mismatched_lengths(
        self, trained: ProductQuantizer, vectors: np.ndarray
    ) -> None:
        tables = trained.distance_tables(vectors[:2])
        with pytest.raises(ValueError, match="same candidates"):
            trained.asymmetric_distances_grouped(
                tables, np.zeros((5, trained.m), np.uint8), np.zeros(4, np.int64)
            )
