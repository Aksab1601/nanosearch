"""The distance primitives everything else is built on.

These tests matter more than their simplicity suggests: the fast expansion of
``||q - x||^2`` is algebraically equivalent to the naive form but not
numerically identical, and a sign error in it would show up as mysteriously bad
recall three modules away rather than as an obvious failure here.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanosearch.metrics import (
    as_float32,
    cosine_from_sqdist,
    l2_normalize,
    squared_l2,
    squared_norms,
    topk_smallest,
)


def naive_squared_l2(queries: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """The obvious O(nq * n * d) implementation, as a reference."""
    return np.array(
        [[float(np.sum((q - x) ** 2)) for x in vectors] for q in queries], dtype=np.float64
    )


class TestSquaredL2:
    def test_matches_naive_implementation(self, rng: np.random.Generator) -> None:
        q = rng.normal(size=(5, 16)).astype(np.float32)
        x = rng.normal(size=(23, 16)).astype(np.float32)
        np.testing.assert_allclose(squared_l2(q, x), naive_squared_l2(q, x), atol=1e-4)

    def test_distance_to_self_is_zero(self, rng: np.random.Generator) -> None:
        x = rng.normal(size=(12, 8)).astype(np.float32)
        diagonal = np.diag(squared_l2(x, x))
        np.testing.assert_allclose(diagonal, np.zeros(12), atol=1e-5)

    def test_never_negative(self, rng: np.random.Generator) -> None:
        # Floating-point cancellation in the expansion can produce small
        # negatives for near-identical vectors; they must be clamped, or a
        # downstream sqrt returns NaN.
        x = np.full((4, 6), 0.3, dtype=np.float32)
        assert (squared_l2(x, x) >= 0).all()

    def test_precomputed_norms_agree(self, rng: np.random.Generator) -> None:
        q = rng.normal(size=(3, 10)).astype(np.float32)
        x = rng.normal(size=(17, 10)).astype(np.float32)
        np.testing.assert_allclose(
            squared_l2(q, x), squared_l2(q, x, vector_sqnorms=squared_norms(x)), atol=1e-5
        )

    def test_accepts_one_dimensional_query(self, rng: np.random.Generator) -> None:
        x = rng.normal(size=(5, 4)).astype(np.float32)
        assert squared_l2(x[0], x).shape == (1, 5)

    def test_rejects_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            squared_l2(np.zeros((2, 4), np.float32), np.zeros((3, 5), np.float32))

    def test_rejects_wrong_length_norms(self) -> None:
        with pytest.raises(ValueError, match="vector_sqnorms"):
            squared_l2(
                np.zeros((2, 4), np.float32),
                np.zeros((3, 4), np.float32),
                vector_sqnorms=np.zeros(2, np.float32),
            )


class TestNormalisation:
    def test_rows_become_unit_length(self, rng: np.random.Generator) -> None:
        x = rng.normal(size=(9, 7)).astype(np.float32) * 13.0
        np.testing.assert_allclose(np.linalg.norm(l2_normalize(x), axis=1), 1.0, atol=1e-5)

    def test_zero_rows_survive(self) -> None:
        x = np.zeros((2, 5), dtype=np.float32)
        assert np.isfinite(l2_normalize(x)).all()

    def test_cosine_identity_holds(self, rng: np.random.Generator) -> None:
        # The identity the whole metric="cosine" code path relies on:
        # ||q - x||^2 = 2 - 2cos(q, x) for unit vectors.
        q = l2_normalize(rng.normal(size=(4, 16)).astype(np.float32))
        x = l2_normalize(rng.normal(size=(11, 16)).astype(np.float32))
        np.testing.assert_allclose(cosine_from_sqdist(squared_l2(q, x)), q @ x.T, atol=1e-5)

    def test_as_float32_rejects_3d(self) -> None:
        with pytest.raises(ValueError, match="1-D or 2-D"):
            as_float32(np.zeros((2, 2, 2)))


class TestTopK:
    def test_agrees_with_full_sort(self, rng: np.random.Generator) -> None:
        values = rng.normal(size=(6, 50)).astype(np.float32)
        idx, vals = topk_smallest(values, 7)
        expected = np.sort(values, axis=1)[:, :7]
        np.testing.assert_allclose(vals, expected, atol=0)
        np.testing.assert_allclose(np.take_along_axis(values, idx, axis=1), expected, atol=0)

    def test_results_are_sorted_ascending(self, rng: np.random.Generator) -> None:
        _, vals = topk_smallest(rng.normal(size=(3, 40)).astype(np.float32), 9)
        assert (np.diff(vals, axis=1) >= 0).all()

    def test_k_larger_than_available_is_clamped(self) -> None:
        idx, vals = topk_smallest(np.arange(4, dtype=np.float32), 10)
        assert idx.shape == (4,) and vals.shape == (4,)

    def test_k_equal_to_n(self) -> None:
        idx, _ = topk_smallest(np.array([3.0, 1.0, 2.0], dtype=np.float32), 3)
        np.testing.assert_array_equal(idx, [1, 2, 0])

    def test_rejects_non_positive_k(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            topk_smallest(np.zeros(5, np.float32), 0)
