"""Saving and loading indexes.

Two properties are tested for every backend:

* a reloaded index returns *identical* results, not merely similar ones
* nothing is pickled, so loading a file someone else produced cannot execute
  their code -- the reason the format is ``.npz`` plus a JSON sidecar rather
  than the one-line ``pickle.dump`` that would otherwise be tempting
"""

from __future__ import annotations

import numpy as np
import pytest

from nanosearch import FlatIndex, IVFIndex, IVFPQIndex, load_index
from nanosearch.base import BaseIndex


def _build(kind: str, vectors: np.ndarray) -> BaseIndex:
    dim = vectors.shape[1]
    if kind == "flat":
        index: BaseIndex = FlatIndex(dim=dim, metric="cosine")
    elif kind == "ivf":
        index = IVFIndex(dim=dim, nlist=32, metric="cosine", nprobe=4, seed=0)
    else:
        index = IVFPQIndex(dim=dim, nlist=32, m=8, metric="cosine", nprobe=4, seed=0)
    index.train(vectors)
    index.add(vectors)
    return index


@pytest.mark.parametrize("kind", ["flat", "ivf", "ivfpq"])
class TestRoundTrip:
    def test_results_are_identical_after_reload(
        self, kind: str, vectors: np.ndarray, queries: np.ndarray, tmp_path
    ) -> None:
        index = _build(kind, vectors)
        before_ids, before_scores = index.search(queries[:20], k=10)

        path = index.save(tmp_path / f"{kind}.npz")
        restored = load_index(path)
        after_ids, after_scores = restored.search(queries[:20], k=10)

        np.testing.assert_array_equal(after_ids, before_ids)
        np.testing.assert_allclose(after_scores, before_scores, atol=0)

    def test_configuration_survives(self, kind: str, vectors: np.ndarray, tmp_path) -> None:
        index = _build(kind, vectors)
        restored = load_index(index.save(tmp_path / f"{kind}.npz"))
        assert restored.stats() == index.stats()
        assert restored.ntotal == index.ntotal
        assert restored.dim == index.dim
        assert restored.metric == index.metric

    def test_file_contains_no_pickles(self, kind: str, vectors: np.ndarray, tmp_path) -> None:
        index = _build(kind, vectors)
        path = index.save(tmp_path / f"{kind}.npz")
        # allow_pickle=False is the assertion: it raises if any array needs it.
        with np.load(path, allow_pickle=False) as handle:
            assert "__meta__" in handle.files
            for name in handle.files:
                assert handle[name] is not None

    def test_custom_ids_survive(self, kind: str, vectors: np.ndarray, tmp_path) -> None:
        dim = vectors.shape[1]
        index = _build(kind, vectors)
        fresh = type(index)(dim=dim) if kind == "flat" else None
        if fresh is None:
            pytest.skip("id handling is covered by the flat case and shared in BaseIndex")
        fresh.add(vectors[:10], ids=np.arange(1000, 1010))
        restored = load_index(fresh.save(tmp_path / "ids.npz"))
        ids, _ = restored.search(vectors[3:4], k=1)
        assert ids[0, 0] == 1003

    def test_loaded_index_accepts_more_vectors(
        self, kind: str, vectors: np.ndarray, tmp_path
    ) -> None:
        index = _build(kind, vectors[:2000])
        restored = load_index(index.save(tmp_path / f"{kind}.npz"))
        added = restored.add(vectors[2000:2100])
        assert restored.ntotal == 2100
        assert len(added) == 100


class TestDispatch:
    def test_load_recovers_the_right_class(self, vectors: np.ndarray, tmp_path) -> None:
        for kind, expected in [
            ("flat", FlatIndex),
            ("ivf", IVFIndex),
            ("ivfpq", IVFPQIndex),
        ]:
            index = _build(kind, vectors[:1500])
            restored = load_index(index.save(tmp_path / f"dispatch_{kind}.npz"))
            assert isinstance(restored, expected)

    def test_rejects_an_unknown_class_name(self, tmp_path) -> None:
        path = tmp_path / "bogus.npz"
        np.savez(path, __meta__=np.asarray('{"__class__": "QuantumIndex"}'))
        with pytest.raises(ValueError, match="unknown index class"):
            load_index(path)
