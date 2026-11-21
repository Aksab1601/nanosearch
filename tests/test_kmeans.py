"""k-means: the quantiser both IVF and PQ are built on."""

from __future__ import annotations

import numpy as np
import pytest

from nanosearch.kmeans import KMeans
from nanosearch.metrics import squared_l2


class TestFit:
    def test_learns_requested_number_of_centroids(self, vectors: np.ndarray) -> None:
        km = KMeans(n_clusters=16, seed=0).fit(vectors)
        assert km.centroids_.shape == (16, vectors.shape[1])
        assert km.centroids_.dtype == np.float32

    def test_recovers_well_separated_clusters(self) -> None:
        # Four blobs far apart: every centroid should land inside one of them,
        # so inertia is governed by the blob spread rather than by the gaps.
        generator = np.random.default_rng(3)
        centers = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=np.float32)
        data = np.repeat(centers, 120, axis=0) + generator.normal(scale=0.3, size=(480, 2)).astype(
            np.float32
        )

        km = KMeans(n_clusters=4, seed=0).fit(data)
        for center in centers:
            nearest = np.min(np.sum((km.centroids_ - center) ** 2, axis=1))
            assert nearest < 1.0

    def test_inertia_beats_random_centroids(self, vectors: np.ndarray) -> None:
        km = KMeans(n_clusters=12, seed=0).fit(vectors)
        picks = np.random.default_rng(0).choice(vectors.shape[0], size=12, replace=False)
        random_inertia = float(squared_l2(vectors, vectors[picks]).min(axis=1).sum())
        assert km.inertia_ < random_inertia

    def test_is_reproducible_for_one_seed(self, vectors: np.ndarray) -> None:
        first = KMeans(n_clusters=8, seed=42).fit(vectors).centroids_
        second = KMeans(n_clusters=8, seed=42).fit(vectors).centroids_
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_explore_differently(self, vectors: np.ndarray) -> None:
        first = KMeans(n_clusters=8, seed=1).fit(vectors).centroids_
        second = KMeans(n_clusters=8, seed=2).fit(vectors).centroids_
        assert not np.array_equal(first, second)

    def test_stops_early_when_converged(self) -> None:
        # Identical points converge immediately: the tolerance check should cut
        # the loop short rather than burning all n_iter iterations.
        data = np.tile(np.array([[1.0, 2.0]], dtype=np.float32), (200, 1))
        km = KMeans(n_clusters=2, n_iter=50, seed=0).fit(data)
        assert km.n_iter_ < 50


class TestAssign:
    def test_assigns_each_point_to_its_nearest_centroid(self, vectors: np.ndarray) -> None:
        km = KMeans(n_clusters=10, seed=0).fit(vectors)
        labels = km.assign(vectors)
        brute_force = np.argmin(squared_l2(vectors, km.centroids_), axis=1)
        np.testing.assert_array_equal(labels, brute_force)

    def test_returns_distance_alongside_label(self, vectors: np.ndarray) -> None:
        km = KMeans(n_clusters=6, seed=0).fit(vectors)
        labels, distances = km.assign_with_distance(vectors)
        expected = squared_l2(vectors, km.centroids_)[np.arange(vectors.shape[0]), labels]
        np.testing.assert_allclose(distances, expected, atol=1e-4)

    def test_chunking_does_not_change_assignments(self, vectors: np.ndarray) -> None:
        km = KMeans(n_clusters=6, seed=0).fit(vectors)
        km.chunk_size = 97  # deliberately not a divisor of the row count
        chunked = km.assign(vectors)
        km.chunk_size = 1_000_000
        np.testing.assert_array_equal(chunked, km.assign(vectors))

    def test_rejects_use_before_fit(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            KMeans(n_clusters=3).assign(np.zeros((5, 2), dtype=np.float32))


class TestDegenerateInputs:
    def test_more_clusters_than_points(self) -> None:
        data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        km = KMeans(n_clusters=8, seed=0).fit(data)
        assert km.centroids_.shape == (8, 2)
        assert np.isfinite(km.centroids_).all()

    def test_empty_clusters_are_respawned_not_left_dead(self) -> None:
        # Two tight groups but ten centroids: some centroids will end up
        # unclaimed, and an unclaimed centroid must be moved rather than left
        # at the origin where it would distort nothing but occupy a slot.
        generator = np.random.default_rng(5)
        data = np.vstack(
            [
                generator.normal(loc=0.0, scale=0.01, size=(100, 4)),
                generator.normal(loc=50.0, scale=0.01, size=(100, 4)),
            ]
        ).astype(np.float32)
        km = KMeans(n_clusters=10, seed=0).fit(data)
        assert np.isfinite(km.centroids_).all()
        # Every centroid should sit near actual data, not in the void between.
        for centroid in km.centroids_:
            assert np.min(np.sum((data - centroid) ** 2, axis=1)) < 100.0

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            KMeans(n_clusters=2).fit(np.zeros((0, 3), dtype=np.float32))

    def test_rejects_non_positive_clusters(self) -> None:
        with pytest.raises(ValueError, match="n_clusters must be positive"):
            KMeans(n_clusters=0).fit(np.zeros((5, 2), dtype=np.float32))

    def test_subsampling_caps_training_set(self, vectors: np.ndarray) -> None:
        # max_points_per_centroid bounds build time on large corpora; the
        # result must still be a usable quantiser.
        km = KMeans(n_clusters=4, max_points_per_centroid=5, seed=0).fit(vectors)
        assert km.centroids_.shape == (4, vectors.shape[1])
        assert np.isfinite(km.centroids_).all()
