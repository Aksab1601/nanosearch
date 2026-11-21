"""The index interface every backend implements.

Three backends ship with nanosearch -- ``FlatIndex`` (exact), ``IVFIndex``
(approximate, full-precision) and ``IVFPQIndex`` (approximate, compressed) --
and the only reason to write them against a shared interface is so the
benchmark harness, the service layer and the tests can swap one for another
without changing a line. That is also what makes the recall-vs-latency
comparison in the README an apples-to-apples one.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import FloatArray, IntArray, as_float32, l2_normalize

__all__ = ["BaseIndex", "SearchResult"]

Metric = str  # "cosine" | "l2"


class SearchResult:
    """Ids and scores for one batch of queries.

    ``ids`` and ``scores`` are both ``(n_queries, k)``. A row is padded with
    ``-1`` ids and ``inf``/``-inf`` scores when fewer than ``k`` candidates
    were found, which happens with an aggressive ``nprobe`` on a small index.
    Padding rather than ragged output keeps downstream array code simple.
    """

    __slots__ = ("ids", "scores")

    def __init__(self, ids: IntArray, scores: FloatArray) -> None:
        self.ids = ids
        self.scores = scores

    def __iter__(self) -> Any:
        """Support ``ids, scores = index.search(...)``."""
        return iter((self.ids, self.scores))

    def __getitem__(self, item: int) -> Any:
        """Support ``index.search(...)[0]`` for the ids alone."""
        return (self.ids, self.scores)[item]

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f"SearchResult(ids={self.ids.shape}, scores={self.scores.shape})"


class BaseIndex(ABC):
    """Common behaviour: metric handling, id bookkeeping, persistence."""

    def __init__(self, dim: int, metric: Metric = "cosine") -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if metric not in ("cosine", "l2"):
            raise ValueError(f"metric must be 'cosine' or 'l2', got {metric!r}")
        self.dim = int(dim)
        self.metric = metric
        # External ids live in a parallel array: internal positions are dense
        # 0..ntotal-1 (fast array maths), user ids can be anything.
        self._ids: IntArray = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------ properties

    @property
    def ntotal(self) -> int:
        """Number of vectors currently in the index."""
        return int(self._ids.shape[0])

    @property
    def is_trained(self) -> bool:
        """Whether the index is ready to accept vectors.

        Exact search needs no training. Anything that partitions or compresses
        the space has to see a sample of the data first.
        """
        return True

    # --------------------------------------------------------------- helpers

    def _prepare(self, vectors: Any) -> FloatArray:
        """Validate input and apply the metric's normalisation convention."""
        arr = as_float32(vectors)
        if arr.shape[1] != self.dim:
            raise ValueError(f"index is {self.dim}-d, got {arr.shape[1]}-d vectors")
        if not np.isfinite(arr).all():
            raise ValueError("vectors contain NaN or inf")
        return l2_normalize(arr) if self.metric == "cosine" else arr

    def _next_ids(self, n: int, ids: Any | None) -> IntArray:
        if ids is None:
            start = int(self._ids.max()) + 1 if self._ids.size else 0
            return np.arange(start, start + n, dtype=np.int64)
        arr = np.asarray(ids, dtype=np.int64).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"got {arr.shape[0]} ids for {n} vectors")
        return arr

    def _blank(self, nq: int, k: int) -> tuple[IntArray, FloatArray]:
        """Pre-filled result arrays: id ``-1`` and a worst-possible score."""
        ids = np.full((nq, k), -1, dtype=np.int64)
        fill = -np.inf if self.metric == "cosine" else np.inf
        return ids, np.full((nq, k), fill, dtype=np.float32)

    def _to_score(self, sqdist: FloatArray) -> FloatArray:
        """Convert internal squared L2 distances to user-facing scores.

        Cosine returns a similarity in ``[-1, 1]`` where larger is better; L2
        returns a distance where smaller is better. Either way the *ranking*
        was produced by the same squared-L2 code path.
        """
        if self.metric == "cosine":
            return (1.0 - 0.5 * sqdist).astype(np.float32, copy=False)
        return np.sqrt(sqdist, dtype=np.float32)

    # ------------------------------------------------------------- interface

    def train(self, vectors: Any) -> None:  # noqa: ARG002
        """Learn whatever structure the backend needs. No-op for exact search."""
        return

    @abstractmethod
    def add(self, vectors: Any, ids: Any | None = None) -> IntArray:
        """Insert vectors, returning the ids assigned to them."""

    @abstractmethod
    def search(self, queries: Any, k: int = 10, **kwargs: Any) -> SearchResult:
        """Return the ``k`` best matches for each query."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Human-readable description of size, configuration and memory use."""

    @abstractmethod
    def _state(self) -> dict[str, Any]:
        """Arrays and scalars that fully describe this index."""

    @classmethod
    @abstractmethod
    def _from_state(cls, state: dict[str, Any]) -> BaseIndex:
        """Rebuild an index from ``_state`` output."""

    # ----------------------------------------------------------- persistence

    def save(self, path: str | Path) -> Path:
        """Write the index to a single ``.npz`` file.

        ``.npz`` is just a zip of ``.npy`` arrays, so saving is a memory copy
        rather than a serialisation pass, and nothing is pickled -- loading an
        index someone else built cannot execute their code. Scalars travel in a
        JSON sidecar string inside the same archive.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self._state()
        arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
        meta = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
        meta["__class__"] = type(self).__name__
        # Annotated as Any-valued: numpy types savez's **kwds against its own
        # allow_pickle keyword, so a narrowly-typed dict fails the check.
        payload: dict[str, Any] = {"__meta__": np.asarray(json.dumps(meta)), **arrays}
        np.savez(path, **payload)
        return path if path.suffix else path.with_suffix(".npz")

    @staticmethod
    def load(path: str | Path) -> BaseIndex:
        """Load an index saved by :meth:`save`, dispatching on the stored class."""
        from . import flat, ivf, ivfpq

        registry: dict[str, type[BaseIndex]] = {
            "FlatIndex": flat.FlatIndex,
            "IVFIndex": ivf.IVFIndex,
            "IVFPQIndex": ivfpq.IVFPQIndex,
        }
        with np.load(Path(path), allow_pickle=False) as handle:
            meta = json.loads(str(handle["__meta__"]))
            state: dict[str, Any] = {k: handle[k] for k in handle.files if k != "__meta__"}
            cls_name = meta.pop("__class__")
            state.update(meta)
        if cls_name not in registry:
            raise ValueError(f"unknown index class {cls_name!r}")
        return registry[cls_name]._from_state(state)
