"""Inverted lists: the bucket structure shared by the IVF indexes.

An inverted list (the term comes from text search, where it maps a term to the
documents containing it) maps *cluster id* to the positions of the vectors
assigned to that cluster. Searching ``nprobe`` of ``nlist`` clusters means
touching roughly ``nprobe / nlist`` of the database -- that ratio is the entire
speed-up, and the quality of the clustering is what decides how much recall it
costs.

Storage is a list of ``int64`` arrays while the index is being built (cheap
appends) and CSR -- one flat array plus ``nlist + 1`` offsets -- when
serialised, which is both compact and exactly how production indexes lay this
out in memory.
"""

from __future__ import annotations

import numpy as np

from .metrics import IntArray

__all__ = ["InvertedLists"]


class InvertedLists:
    """``nlist`` growable buckets of vector positions."""

    __slots__ = ("nlist", "_lists")

    def __init__(self, nlist: int) -> None:
        if nlist <= 0:
            raise ValueError(f"nlist must be positive, got {nlist}")
        self.nlist = int(nlist)
        self._lists: list[IntArray] = [np.zeros(0, dtype=np.int64) for _ in range(self.nlist)]

    def __len__(self) -> int:
        return self.nlist

    def __getitem__(self, list_id: int) -> IntArray:
        return self._lists[list_id]

    # -------------------------------------------------------------- mutation

    def extend(self, labels: IntArray, positions: IntArray) -> None:
        """Append ``positions`` to the buckets named by ``labels``.

        Grouping by label first means one concatenation per *touched bucket*
        instead of one per vector -- the difference between a bulk load that
        takes a second and one that takes minutes.
        """
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        if labels.shape != positions.shape:
            raise ValueError("labels and positions must have the same length")
        if labels.size == 0:
            return
        if labels.min() < 0 or labels.max() >= self.nlist:
            raise ValueError("label out of range for this number of lists")

        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        sorted_positions = positions[order]
        unique, starts = np.unique(sorted_labels, return_index=True)
        bounds = np.append(starts, sorted_labels.shape[0])

        for i, list_id in enumerate(unique):
            chunk = sorted_positions[bounds[i] : bounds[i + 1]]
            current = self._lists[int(list_id)]
            self._lists[int(list_id)] = (
                chunk.copy() if current.size == 0 else np.concatenate([current, chunk])
            )

    # ------------------------------------------------------------ inspection

    def sizes(self) -> IntArray:
        return np.array([lst.shape[0] for lst in self._lists], dtype=np.int64)

    def gather(self, list_ids: IntArray) -> IntArray:
        """All positions in the given lists, concatenated."""
        chunks = [self._lists[int(i)] for i in np.asarray(list_ids).reshape(-1)]
        chunks = [c for c in chunks if c.size]
        if not chunks:
            return np.zeros(0, dtype=np.int64)
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def imbalance_factor(self) -> float:
        """1.0 for perfectly even lists, higher when a few buckets dominate.

        Defined as ``n * sum(s_i^2) / (sum s_i)^2``. It matters because search
        cost is driven by the size of the lists a query actually probes, not by
        the average: one bucket holding 20% of the corpus turns every query
        that lands near it into a near-exhaustive scan. Values under ~1.3 are
        healthy; a large value is the signal to retrain with more data or a
        different ``nlist``.
        """
        sizes = self.sizes().astype(np.float64)
        total = sizes.sum()
        if total <= 0:
            return 1.0
        return float(sizes.shape[0] * np.square(sizes).sum() / (total * total))

    # ----------------------------------------------------------- persistence

    def to_csr(self) -> tuple[IntArray, IntArray]:
        """Pack to ``(offsets, flat)`` with ``offsets`` of length ``nlist + 1``."""
        sizes = self.sizes()
        offsets = np.zeros(self.nlist + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        flat = (
            np.concatenate([lst for lst in self._lists if lst.size])
            if offsets[-1] > 0
            else np.zeros(0, dtype=np.int64)
        )
        return offsets, flat

    @classmethod
    def from_csr(cls, offsets: IntArray, flat: IntArray) -> InvertedLists:
        offsets = np.asarray(offsets, dtype=np.int64)
        flat = np.asarray(flat, dtype=np.int64)
        lists = cls(nlist=offsets.shape[0] - 1)
        for i in range(lists.nlist):
            lists._lists[i] = flat[offsets[i] : offsets[i + 1]].copy()
        return lists
