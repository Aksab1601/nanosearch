"""Two-stage retrieval: compressed shortlist in RAM, exact re-rank from disk.

The problem with PQ
-------------------
Compression costs accuracy, and past a point more ``nprobe`` cannot buy it
back: once the right cell is being probed, what limits recall is that the
*codes* are approximate. The recall ceiling of an IVF-PQ index is a property
of ``m``, not of how hard you search.

The fix every production system uses
------------------------------------
Treat the compressed index as a *candidate generator*, not as the final
ranking. Ask it for ``k * factor`` candidates, then score just those few
hundred against full-precision vectors and keep the real top-``k``. The
expensive exact distance is paid on hundreds of vectors instead of millions.

Where the full vectors live is the interesting part. Keeping them in RAM would
undo the memory saving, so here they sit in a flat float32 file read through
``np.memmap``: the OS pages in only the rows actually touched, roughly
``k * factor`` random reads of ``4 * dim`` bytes per query. RAM holds the
``m``-byte codes; the cold bytes stay on disk. This is the same tiering
FAISS's ``IndexRefineFlat`` and every serious vector database implement, and
it is why "approximate index" rarely means "approximate answers" in practice.

Two costs, stated plainly: disk reads add latency (a few hundred microseconds
on an SSD, far more on spinning disk or network storage), and the vectors have
to be stored somewhere, so total footprint is unchanged -- what changes is how
much has to be in *memory*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseIndex, Metric, SearchResult
from .metrics import FloatArray, IntArray, squared_l2, squared_norms, topk_smallest

__all__ = ["RefineIndex"]


class RefineIndex(BaseIndex):
    """Wraps any index with an exact re-ranking stage backed by a memmap file.

    Parameters
    ----------
    base:
        The candidate generator, typically an :class:`~nanosearch.IVFPQIndex`.
        Its own ids are overwritten: the wrapper hands it dense positions so a
        returned candidate indexes straight into the vector file.
    path:
        Where the float32 vectors are written. Reopened read-only as a memmap
        after each ``add``.
    factor:
        Shortlist multiplier. ``k * factor`` candidates are re-ranked;
        4 to 10 is the usual range. Larger closes the recall gap and costs
        more disk reads.
    """

    def __init__(
        self,
        base: BaseIndex,
        path: str | Path,
        factor: int = 4,
        metric: Metric | None = None,
    ) -> None:
        super().__init__(base.dim, metric or base.metric)
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")
        self.base = base
        self.factor = int(factor)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_bytes(b"")
        self._memmap: np.memmap | None = None
        self._sqnorms: FloatArray = np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------ properties

    @property
    def is_trained(self) -> bool:
        return self.base.is_trained

    def train(self, vectors: Any) -> None:
        self.base.train(vectors)

    # ------------------------------------------------------------------- add

    def add(self, vectors: Any, ids: Any | None = None) -> IntArray:
        prepared = self._prepare(vectors)
        n = prepared.shape[0]
        assigned = self._next_ids(n, ids)
        positions = np.arange(self.ntotal, self.ntotal + n, dtype=np.int64)

        # Append raw float32 rows. A plain append is cheaper than growing a
        # memmap, and the file layout is trivially predictable: row i starts at
        # byte i * 4 * dim.
        with self.path.open("ab") as handle:
            handle.write(np.ascontiguousarray(prepared, dtype=np.float32).tobytes())

        self.base.add(prepared, ids=positions)
        self._ids = np.concatenate([self._ids, assigned])
        self._sqnorms = np.concatenate([self._sqnorms, squared_norms(prepared)])
        self._memmap = None  # invalidate; reopened lazily at the new length
        return assigned

    # ---------------------------------------------------------------- search

    def _vectors(self) -> np.memmap:
        if self._memmap is None:
            self._memmap = np.memmap(
                self.path, dtype=np.float32, mode="r", shape=(self.ntotal, self.dim)
            )
        return self._memmap

    def search(
        self,
        queries: Any,
        k: int = 10,
        factor: int | None = None,
        **kwargs: Any,
    ) -> SearchResult:
        q = self._prepare(queries)
        nq = q.shape[0]
        ids, scores = self._blank(nq, k)
        if self.ntotal == 0:
            return SearchResult(ids, scores)

        mult = int(factor or self.factor)
        shortlist_size = min(max(k * mult, k), self.ntotal)
        candidates, _ = self.base.search(q, k=shortlist_size, **kwargs)
        vectors = self._vectors()

        for i in range(nq):
            # Base ids are positions; -1 is padding from an under-full shortlist.
            positions = candidates[i][candidates[i] >= 0]
            if positions.size == 0:
                continue
            # Sorting makes the memmap reads sequential-ish, which matters more
            # than it looks: random 4*dim-byte reads defeat OS readahead.
            positions = np.unique(positions)
            rows = np.asarray(vectors[positions], dtype=np.float32)
            dist = squared_l2(q[i : i + 1], rows, vector_sqnorms=self._sqnorms[positions])[0]
            take = min(k, positions.shape[0])
            sel, vals = topk_smallest(dist, take)
            ids[i, :take] = self._ids[positions[sel]]
            scores[i, :take] = self._to_score(vals)

        return SearchResult(ids, scores)

    # ----------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        base_stats = self.base.stats()
        disk_bytes = self.ntotal * self.dim * 4
        return {
            **base_stats,
            "index_type": f"refine({base_stats['index_type']})",
            "factor": self.factor,
            "resident_memory_mb": base_stats.get("memory_mb", 0.0),
            "disk_mb": round(disk_bytes / 1e6, 3),
            "vector_file": str(self.path),
            "exact": False,
        }

    # ----------------------------------------------------------- persistence

    def _state(self) -> dict[str, Any]:
        raise NotImplementedError(
            "RefineIndex is not saved directly: save the base index with "
            "base.save(...) and keep the vector file beside it, then rebuild "
            "the wrapper with RefineIndex(load_index(...), path=...)."
        )

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> RefineIndex:
        raise NotImplementedError
