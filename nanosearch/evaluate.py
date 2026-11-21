"""Measuring quality and speed.

An approximate index without a recall number attached is a liability: it is
trivial to build something fast that returns wrong answers. These helpers are
what every claim in the README is produced by, and they are deliberately part
of the library rather than the benchmark scripts so tests can assert on them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .metrics import IntArray

__all__ = ["LatencyReport", "measure_latency", "recall_at_k"]


def recall_at_k(predicted: IntArray, truth: IntArray, k: int | None = None) -> float:
    """Fraction of the exact top-``k`` found anywhere in what the index returned.

    ``recall@k`` is the standard ANN quality metric. Set overlap is the right
    notion because a vector store almost always feeds a downstream re-ranker
    or an LLM context window, which cares about *which* documents arrived, not
    the order they arrived in.

    ``k`` controls how deep into ``truth`` to look; the whole ``predicted`` row
    counts as a hit, which makes two different questions expressible with one
    function:

    * ``recall_at_k(pred10, gold10)`` -- classic recall@10: did the top-10
      returned contain the true top-10
    * ``recall_at_k(pred100, gold10, k=10)`` -- recall@10 within a 100-item
      shortlist: the number that matters when a re-ranker or an LLM sees the
      whole shortlist, and the number production systems actually tune

    Both arrays are ``(n_queries, width)`` id matrices; ``-1`` marks padding
    and is ignored.
    """
    pred = np.asarray(predicted, dtype=np.int64)
    gold = np.asarray(truth, dtype=np.int64)
    if pred.ndim != 2 or gold.ndim != 2:
        raise ValueError("predicted and truth must both be 2-D id matrices")
    if pred.shape[0] != gold.shape[0]:
        raise ValueError("predicted and truth must cover the same queries")

    k_eff = min(k or gold.shape[1], gold.shape[1])
    if k_eff <= 0:
        raise ValueError("k must be positive")

    hits = 0
    denom = 0
    for row_pred, row_gold in zip(pred, gold[:, :k_eff], strict=False):
        wanted = {int(v) for v in row_gold if v >= 0}
        if not wanted:
            continue
        got = {int(v) for v in row_pred if v >= 0}
        hits += len(wanted & got)
        denom += len(wanted)
    return hits / denom if denom else 0.0


@dataclass
class LatencyReport:
    """Latency summary for one search configuration."""

    n_queries: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_latency(
    search: Callable[[np.ndarray], Any],
    queries: np.ndarray,
    warmup: int = 5,
) -> LatencyReport:
    """Time ``search`` one query at a time and report the tail, not just the mean.

    Queries are issued singly because that is how a search service is actually
    hit, and because batching hides exactly the Python-level overhead that
    matters at serving time. Percentiles are reported because a p95 three times
    the median is a real problem -- for IVF it usually means a few queries are
    landing on oversized cells (see ``imbalance_factor``) -- and a mean would
    never show it.

    ``warmup`` iterations are discarded so that first-call allocation and BLAS
    thread spin-up do not land in the measurement.
    """
    q = np.asarray(queries, dtype=np.float32)
    if q.ndim != 2 or q.shape[0] == 0:
        raise ValueError("queries must be a non-empty 2-D array")

    for i in range(min(warmup, q.shape[0])):
        search(q[i : i + 1])

    timings = np.empty(q.shape[0], dtype=np.float64)
    for i in range(q.shape[0]):
        start = time.perf_counter()
        search(q[i : i + 1])
        timings[i] = time.perf_counter() - start

    ms = timings * 1e3
    return LatencyReport(
        n_queries=int(q.shape[0]),
        mean_ms=round(float(ms.mean()), 4),
        p50_ms=round(float(np.percentile(ms, 50)), 4),
        p95_ms=round(float(np.percentile(ms, 95)), 4),
        p99_ms=round(float(np.percentile(ms, 99)), 4),
        qps=round(float(1.0 / timings.mean()), 1),
    )
