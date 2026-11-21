"""A guided tour: text search, then the recall/latency/memory trade-off.

    python examples/demo.py

Runs in about half a minute and prints the two things worth seeing -- that
semantic search over the bundled corpus works, and what each index backend
actually costs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from datasets import make_embedding_like, split_queries  # noqa: E402

from nanosearch import (  # noqa: E402
    FlatIndex,
    IVFIndex,
    IVFPQIndex,
    RefineIndex,
    measure_latency,
    recall_at_k,
)
from nanosearch.engine import SearchEngine  # noqa: E402
from nanosearch.store import DocumentStore  # noqa: E402

CORPUS = Path(__file__).resolve().parent.parent / "nanosearch" / "data" / "corpus.jsonl"

QUERIES = [
    "how does a model avoid memorising its training data",
    "my database queries got slow after adding indexes",
    "what happens to a star when it runs out of fuel",
    "my tomato plants are not producing fruit",
    "how do I stop a sauce from splitting",
]


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def text_search_demo() -> None:
    rule("1. Semantic search over the bundled corpus")
    texts, metadatas = DocumentStore.from_jsonl(CORPUS)

    started = time.perf_counter()
    engine = SearchEngine.build(texts, metadatas=metadatas)
    print(
        f"indexed {len(texts)} documents in {time.perf_counter() - started:.2f}s "
        f"({engine.embedder.vocabulary_size} terms -> {engine.embedder.dim} dimensions, "
        f"{engine.index.stats()['index_type']} index)"
    )

    for query in QUERIES:
        hits = engine.search(query, k=2)
        print(f'\n  "{query}"')
        for hit in hits:
            topic = hit.metadata.get("topic", "?")
            print(f"    {hit.score:+.3f}  [{topic:16s}] {hit.text[:68]}...")


def index_comparison_demo() -> None:
    rule("2. What each index costs (50k x 128-d, embedding-like vectors)")

    data = make_embedding_like(n=50_300, dim=128, n_clusters=64, seed=0)
    database, queries = split_queries(data, 300, seed=1)
    dim = database.shape[1]

    exact = FlatIndex(dim=dim, metric="cosine")
    exact.add(database)
    gold = exact.search(queries, k=100)[0]
    exact_latency = measure_latency(lambda q: exact.search(q, k=10), queries[:100])

    header = f"{'index':<30} {'recall@10':>10} {'in top-100':>11} {'p50 ms':>8} {'vs exact':>9} {'RAM MB':>8}"
    print(f"\n{header}\n{'=' * len(header)}")
    print(
        f"{'flat (exact)':<30} {1.0:>10.4f} {1.0:>11.4f} "
        f"{exact_latency.p50_ms:>8.3f} {'1.0x':>9} {exact.stats()['memory_mb']:>8.1f}"
    )

    ivf = IVFIndex(dim=dim, nlist=224, metric="cosine", seed=0)
    ivf.train(database)
    ivf.add(database)
    for nprobe in (1, 4, 16):
        latency = measure_latency(
            lambda q, nprobe=nprobe: ivf.search(q, k=10, nprobe=nprobe), queries[:100]
        )
        recall = recall_at_k(ivf.search(queries, k=10, nprobe=nprobe)[0], gold[:, :10])
        deep = recall_at_k(ivf.search(queries, k=100, nprobe=nprobe)[0], gold[:, :10], k=10)
        print(
            f"{f'ivf nprobe={nprobe}':<30} {recall:>10.4f} {deep:>11.4f} "
            f"{latency.p50_ms:>8.3f} {f'{exact_latency.p50_ms / latency.p50_ms:.1f}x':>9} "
            f"{ivf.stats()['memory_mb']:>8.1f}"
        )

    ivfpq = IVFPQIndex(dim=dim, nlist=224, m=32, metric="cosine", seed=0)
    ivfpq.train(database)
    ivfpq.add(database)
    for nprobe in (4, 16):
        latency = measure_latency(
            lambda q, nprobe=nprobe: ivfpq.search(q, k=10, nprobe=nprobe), queries[:100]
        )
        recall = recall_at_k(ivfpq.search(queries, k=10, nprobe=nprobe)[0], gold[:, :10])
        deep = recall_at_k(ivfpq.search(queries, k=100, nprobe=nprobe)[0], gold[:, :10], k=10)
        print(
            f"{f'ivfpq m=32 nprobe={nprobe}':<30} {recall:>10.4f} {deep:>11.4f} "
            f"{latency.p50_ms:>8.3f} {f'{exact_latency.p50_ms / latency.p50_ms:.1f}x':>9} "
            f"{ivfpq.stats()['memory_mb']:>8.1f}"
        )

    vector_file = Path("/tmp/nanosearch_demo_vectors.f32")
    vector_file.unlink(missing_ok=True)
    refine = RefineIndex(ivfpq.clone_trained(), path=vector_file, factor=16)
    refine.add(database)
    for factor in (8, 32):
        latency = measure_latency(
            lambda q, factor=factor: refine.search(q, k=10, factor=factor, nprobe=8), queries[:100]
        )
        recall = recall_at_k(refine.search(queries, k=10, factor=factor, nprobe=8)[0], gold[:, :10])
        print(
            f"{f'two-stage factor={factor}':<30} {recall:>10.4f} {'(exact)':>11} "
            f"{latency.p50_ms:>8.3f} {f'{exact_latency.p50_ms / latency.p50_ms:.1f}x':>9} "
            f"{refine.stats()['resident_memory_mb']:>8.1f}"
        )
    print(
        f"\n  Two-stage keeps {refine.stats()['resident_memory_mb']:.1f} MB of codes in RAM and "
        f"reads the {refine.stats()['disk_mb']:.0f} MB of vectors from disk only for the "
        f"shortlist it re-ranks."
    )
    vector_file.unlink(missing_ok=True)


def persistence_demo() -> None:
    rule("3. Saving and loading")
    texts, metadatas = DocumentStore.from_jsonl(CORPUS)
    engine = SearchEngine.build(texts, metadatas=metadatas)
    target = Path("/tmp/nanosearch_demo_index")
    engine.save(target)
    restored = SearchEngine.load(target)

    query = "how do I thicken a sauce"
    before = [hit.id for hit in engine.search(query, k=5)]
    after = [hit.id for hit in restored.search(query, k=5)]
    print(f"  saved to {target}")
    print(f"  files: {', '.join(sorted(p.name for p in target.iterdir()))}")
    print(f"  identical results after reload: {before == after}")


def main() -> int:
    np.set_printoptions(precision=4, suppress=True)
    text_search_demo()
    index_comparison_demo()
    persistence_demo()
    print("\nTry the service next:  make serve   (then open http://127.0.0.1:8000)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
