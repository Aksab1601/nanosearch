"""Recall/latency/memory sweep: the numbers quoted in the README come from here.

Run::

    python benchmarks/bench_sweep.py --n 200000 --dim 128 --nlist 1024 --m 16
    python benchmarks/bench_sweep.py --data my_embeddings.npy --nlist 1024

Method
------
* ground truth is exact search on the same data, so recall is measured, never
  estimated
* queries are held out of the database (see ``datasets.split_queries``)
* latency is measured one query at a time, which is how a service is hit, and
  reported as p50/p95 rather than a mean
* each index is built, measured and freed before the next one, so a large run
  never holds two copies of the corpus in memory

Output is a JSON record plus a Markdown table, both written to ``--out``.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_or_make, split_queries  # noqa: E402

from nanosearch import (  # noqa: E402
    FlatIndex,
    IVFIndex,
    IVFPQIndex,
    RefineIndex,
    measure_latency,
    recall_at_k,
)


@dataclass
class Sweep:
    """Collected rows of one benchmark run."""

    meta: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, **row: Any) -> None:
        self.rows.append(row)
        label = row.get("config", "")
        print(
            f"  {label:<34} recall@{row.get('k', 10)}={row.get('recall', float('nan')):.4f}"
            f"  shortlist={row.get('recall_shortlist', float('nan')):.4f}"
            f"  p50={row.get('p50_ms', float('nan')):.3f}ms"
            f"  p95={row.get('p95_ms', float('nan')):.3f}ms"
            f"  qps={row.get('qps', 0):.0f}"
            f"  mem={row.get('memory_mb', 0):.1f}MB",
            flush=True,
        )


def _measure(
    index: Any,
    queries: np.ndarray,
    gold: np.ndarray,
    k: int,
    shortlist: int,
    n_latency: int,
    **search_kwargs: Any,
) -> dict[str, Any]:
    """Recall at ``k``, recall of the true top-``k`` inside a ``shortlist``, and latency."""
    predicted = index.search(queries, k=k, **search_kwargs)[0]
    recall = recall_at_k(predicted, gold, k=k)

    wide = index.search(queries, k=shortlist, **search_kwargs)[0]
    recall_shortlist = recall_at_k(wide, gold, k=k)

    latency = measure_latency(lambda q: index.search(q, k=k, **search_kwargs), queries[:n_latency])
    return {
        "recall": round(recall, 4),
        "recall_shortlist": round(recall_shortlist, 4),
        **latency.as_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None, help="path to a .npy array of vectors")
    parser.add_argument("--n", type=int, default=100_000, help="synthetic corpus size")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--shortlist", type=int, default=100, help="re-ranker candidate depth")
    parser.add_argument("--nlist", type=int, default=0, help="0 selects 4*sqrt(n)")
    parser.add_argument("--m", type=int, default=16, help="PQ bytes per vector")
    parser.add_argument("--nprobe", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--factor", type=int, nargs="*", default=[2, 4, 8, 16])
    parser.add_argument("--kmeans-iter", type=int, default=20)
    parser.add_argument("--latency-queries", type=int, default=120)
    parser.add_argument(
        "--refine-nprobe",
        type=int,
        default=0,
        help="nprobe for the two-stage runs; 0 uses min(max(--nprobe), 16)",
    )
    parser.add_argument("--skip-flat-latency", action="store_true")
    parser.add_argument("--out", type=str, default="docs/bench")
    parser.add_argument("--tag", type=str, default="", help="suffix for the output filenames")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    vectors, provenance = load_or_make(
        args.data, n=args.n + args.queries, dim=args.dim, seed=args.seed
    )
    database, queries = split_queries(vectors, args.queries, seed=args.seed + 1)
    del vectors
    gc.collect()

    n, dim = database.shape
    nlist = args.nlist or max(16, int(4 * np.sqrt(n)))
    print(f"\ndatabase {n} x {dim}  queries {queries.shape[0]}  nlist={nlist}  m={args.m}")
    print(f"data: {provenance}\n", flush=True)

    sweep = Sweep(
        meta={
            "n": int(n),
            "dim": int(dim),
            "n_queries": int(queries.shape[0]),
            "k": args.k,
            "shortlist": args.shortlist,
            "nlist": nlist,
            "m": args.m,
            "data": provenance,
            "kmeans_iter": args.kmeans_iter,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "float32_corpus_mb": round(n * dim * 4 / 1e6, 1),
        }
    )

    # ---------------------------------------------------------------- exact
    print("exact (ground truth):", flush=True)
    t0 = time.perf_counter()
    flat = FlatIndex(dim=dim, metric="cosine")
    flat.add(database)
    build_flat = time.perf_counter() - t0
    gold = flat.search(queries, k=max(args.k, args.shortlist))[0]

    flat_latency = (
        {"p50_ms": float("nan"), "p95_ms": float("nan"), "qps": float("nan")}
        if args.skip_flat_latency
        else measure_latency(
            lambda q: flat.search(q, k=args.k), queries[: args.latency_queries]
        ).as_dict()
    )
    sweep.add(
        config="flat (exact)",
        index="flat",
        k=args.k,
        recall=1.0,
        recall_shortlist=1.0,
        build_s=round(build_flat, 2),
        memory_mb=flat.stats()["memory_mb"],
        bytes_per_vector=flat.stats()["bytes_per_vector"],
        **flat_latency,
    )
    baseline_p50 = flat_latency.get("p50_ms", float("nan"))
    # Released rather than `del`eted: the latency lambda above closes over this
    # name, and deleting a name a closure captures is how that reads as a bug.
    flat = None
    gc.collect()

    # ------------------------------------------------------------------ IVF
    print("\nIVF (full-precision vectors, partitioned):", flush=True)
    t0 = time.perf_counter()
    ivf = IVFIndex(dim=dim, nlist=nlist, metric="cosine", kmeans_iter=args.kmeans_iter)
    ivf.train(database)
    train_ivf = time.perf_counter() - t0
    t0 = time.perf_counter()
    ivf.add(database)
    add_ivf = time.perf_counter() - t0
    ivf_stats = ivf.stats()
    print(
        f"  train={train_ivf:.1f}s add={add_ivf:.1f}s imbalance={ivf_stats['imbalance_factor']} "
        f"list_mean={ivf_stats['list_size_mean']} list_max={ivf_stats['list_size_max']}",
        flush=True,
    )
    for nprobe in args.nprobe:
        if nprobe > nlist:
            continue
        measured = _measure(
            ivf, queries, gold, args.k, args.shortlist, args.latency_queries, nprobe=nprobe
        )
        sweep.add(
            config=f"ivf nprobe={nprobe}",
            index="ivf",
            k=args.k,
            nprobe=nprobe,
            build_s=round(train_ivf + add_ivf, 2),
            memory_mb=ivf_stats["memory_mb"],
            bytes_per_vector=ivf_stats["bytes_per_vector"],
            speedup_vs_exact=round(baseline_p50 / measured["p50_ms"], 2)
            if measured["p50_ms"]
            else None,
            **measured,
        )
    ivf_meta = {
        "imbalance_factor": ivf_stats["imbalance_factor"],
        "list_size_mean": ivf_stats["list_size_mean"],
        "list_size_max": ivf_stats["list_size_max"],
    }
    del ivf
    gc.collect()

    # --------------------------------------------------------------- IVF-PQ
    print(f"\nIVF-PQ (m={args.m} bytes per vector):", flush=True)
    t0 = time.perf_counter()
    ivfpq = IVFPQIndex(
        dim=dim, nlist=nlist, m=args.m, metric="cosine", kmeans_iter=args.kmeans_iter
    )
    ivfpq.train(database)
    train_pq = time.perf_counter() - t0
    t0 = time.perf_counter()
    ivfpq.add(database)
    add_pq = time.perf_counter() - t0
    pq_stats = ivfpq.stats()
    print(
        f"  train={train_pq:.1f}s add={add_pq:.1f}s memory={pq_stats['memory_mb']}MB "
        f"({pq_stats['compression_ratio']}x compression)",
        flush=True,
    )
    for nprobe in args.nprobe:
        if nprobe > nlist:
            continue
        measured = _measure(
            ivfpq, queries, gold, args.k, args.shortlist, args.latency_queries, nprobe=nprobe
        )
        sweep.add(
            config=f"ivfpq m={args.m} nprobe={nprobe}",
            index="ivfpq",
            k=args.k,
            nprobe=nprobe,
            m=args.m,
            build_s=round(train_pq + add_pq, 2),
            memory_mb=pq_stats["memory_mb"],
            bytes_per_vector=pq_stats["bytes_per_vector"],
            compression_ratio=pq_stats["compression_ratio"],
            speedup_vs_exact=round(baseline_p50 / measured["p50_ms"], 2)
            if measured["p50_ms"]
            else None,
            **measured,
        )

    # ------------------------------------------------- IVF-PQ + exact re-rank
    print("\nIVF-PQ + memmap re-rank (two-stage):", flush=True)
    vector_file = Path(args.out) / f"refine_{n}x{dim}.f32"
    vector_file.parent.mkdir(parents=True, exist_ok=True)
    vector_file.unlink(missing_ok=True)
    mid_nprobe = args.refine_nprobe or min(max([p for p in args.nprobe if p <= nlist] or [8]), 16)
    # A *clone* of the trained quantiser, not the populated index: reusing the
    # latter would index every vector twice.
    refine = RefineIndex(ivfpq.clone_trained(), path=vector_file, factor=max(args.factor))
    t0 = time.perf_counter()
    refine.add(database)
    add_refine = time.perf_counter() - t0
    print(
        f"  wrote {vector_file.stat().st_size / 1e6:.0f}MB of vectors in {add_refine:.1f}s",
        flush=True,
    )
    for factor in args.factor:
        measured = _measure(
            refine,
            queries,
            gold,
            args.k,
            args.shortlist,
            args.latency_queries,
            factor=factor,
            nprobe=mid_nprobe,
        )
        sweep.add(
            config=f"refine factor={factor} (nprobe={mid_nprobe})",
            index="refine",
            k=args.k,
            nprobe=mid_nprobe,
            factor=factor,
            m=args.m,
            memory_mb=pq_stats["memory_mb"],
            disk_mb=round(vector_file.stat().st_size / 1e6, 1),
            bytes_per_vector=pq_stats["bytes_per_vector"],
            speedup_vs_exact=round(baseline_p50 / measured["p50_ms"], 2)
            if measured["p50_ms"]
            else None,
            **measured,
        )
    del refine, ivfpq
    gc.collect()
    vector_file.unlink(missing_ok=True)

    # --------------------------------------------------------------- outputs
    sweep.meta.update(ivf_meta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sweep_n{n}_d{dim}_m{args.m}" + (f"_{args.tag}" if args.tag else "")
    (out_dir / f"{stem}.json").write_text(
        json.dumps({"meta": sweep.meta, "rows": sweep.rows}, indent=2)
    )
    (out_dir / f"{stem}.md").write_text(_markdown(sweep))
    print(f"\nwrote {out_dir / f'{stem}.json'} and {out_dir / f'{stem}.md'}")
    return 0


def _markdown(sweep: Sweep) -> str:
    meta = sweep.meta
    lines = [
        f"# Sweep: {meta['n']:,} x {meta['dim']}-d vectors",
        "",
        f"- data: `{meta['data']}`",
        f"- float32 corpus size: {meta['float32_corpus_mb']} MB",
        f"- `nlist` = {meta['nlist']}, PQ `m` = {meta['m']}, k = {meta['k']},"
        f" shortlist = {meta['shortlist']}",
        f"- queries: {meta['n_queries']} held out of the database",
        f"- {meta['platform']}, Python {meta['python']}, NumPy {meta['numpy']}",
        "",
        "| config | recall@k | recall@k in shortlist | p50 ms | p95 ms | QPS |"
        " speedup | RAM MB | bytes/vec |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in sweep.rows:
        speedup = row.get("speedup_vs_exact")
        lines.append(
            "| {config} | {recall:.4f} | {shortlist:.4f} | {p50:.3f} | {p95:.3f} | {qps:.0f} "
            "| {speedup} | {mem:.1f} | {bpv} |".format(
                config=row.get("config", ""),
                recall=row.get("recall", float("nan")),
                shortlist=row.get("recall_shortlist", float("nan")),
                p50=row.get("p50_ms", float("nan")),
                p95=row.get("p95_ms", float("nan")),
                qps=row.get("qps", float("nan")),
                speedup=f"{speedup:.1f}x" if speedup else "1.0x",
                mem=row.get("memory_mb", float("nan")),
                bpv=row.get("bytes_per_vector", ""),
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
