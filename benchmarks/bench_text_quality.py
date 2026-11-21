"""Does the text embedder actually retrieve the right documents?

The index benchmarks answer "did we find the nearest *vectors*". That is a
different question from "did we find the right *documents*", and only the
second one is what a search engine is for. An index with 0.99 recall against
embeddings that do not capture meaning is a fast way to return the wrong
answer.

So: a labelled query set over the demo corpus, two queries per topic, scored by
precision@k -- of the top k documents returned, what fraction belong to the
topic the query is about. The corpus has 8 topics, so a random ranker scores
0.125 and anything near that means the embedder is doing nothing.

Run::

    python benchmarks/bench_text_quality.py
    python benchmarks/bench_text_quality.py --sweep

``--sweep`` grid-searches the embedder's parameters. The defaults in
``nanosearch.text`` were chosen from its output, not by taste.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanosearch.store import DocumentStore  # noqa: E402
from nanosearch.text import TfidfSvdEmbedder  # noqa: E402

CORPUS = Path(__file__).resolve().parent.parent / "nanosearch" / "data" / "corpus.jsonl"

# Deliberately worded as a user would, not as the documents are: no query
# repeats a document's phrasing, so term overlap alone should not be enough.
LABELLED_QUERIES: list[tuple[str, str]] = [
    ("how do neural networks learn from data", "machine-learning"),
    ("stopping a model from memorising the training set", "machine-learning"),
    ("why is my sql query slow", "databases"),
    ("keeping data safe if the server crashes", "databases"),
    ("how far away are distant galaxies", "astronomy"),
    ("what happens when a massive star dies", "astronomy"),
    ("how do I thicken a sauce", "cooking"),
    ("getting a good crust on a steak", "cooking"),
    ("what gearing should I use for steep hills", "cycling"),
    ("riding faster in a group", "cycling"),
    ("which notes make up a major chord", "music"),
    ("why do minor keys sound sad", "music"),
    ("my seedlings keep dying", "gardening"),
    ("improving heavy clay soil", "gardening"),
    ("why is one side of a mountain dry", "geography"),
    ("how are deep coastal inlets formed", "geography"),
]


def evaluate(
    embedder: TfidfSvdEmbedder,
    texts: list[str],
    topics: list[str],
    k: int = 3,
) -> dict[str, float]:
    """Precision@k and mean reciprocal rank over the labelled query set."""
    document_vectors = embedder.encode(texts)
    query_vectors = embedder.encode([query for query, _ in LABELLED_QUERIES])
    # Both sides are L2-normalised, so a dot product *is* cosine similarity.
    similarity = query_vectors @ document_vectors.T

    precision_total = 0.0
    reciprocal_total = 0.0
    for row, (_, expected) in enumerate(LABELLED_QUERIES):
        ranking = np.argsort(-similarity[row])
        top = ranking[:k]
        precision_total += sum(1 for j in top if topics[j] == expected) / k
        first = next(
            (rank for rank, j in enumerate(ranking, start=1) if topics[j] == expected), None
        )
        reciprocal_total += 1.0 / first if first else 0.0

    n = len(LABELLED_QUERIES)
    return {
        f"precision@{k}": round(precision_total / n, 4),
        "mrr": round(reciprocal_total / n, 4),
        "random_baseline": round(1.0 / len(set(topics)), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--sweep", action="store_true", help="grid-search embedder parameters")
    parser.add_argument("--out", type=str, default="docs/bench")
    args = parser.parse_args(argv)

    texts, metadatas = DocumentStore.from_jsonl(CORPUS)
    topics = [str(meta.get("topic", "unknown")) for meta in metadatas]
    print(f"{len(texts)} documents, {len(set(topics))} topics, {len(LABELLED_QUERIES)} queries\n")

    if not args.sweep:
        embedder = TfidfSvdEmbedder(dim=len(texts) // 2).fit(texts)
        scores = evaluate(embedder, texts, topics, k=args.k)
        print(json.dumps(scores, indent=2))
        print(f"\nvocabulary: {embedder.vocabulary_size} terms -> {embedder.dim} dimensions")
        print("\nstrongest terms per dimension (the first few):")
        for component in range(min(4, embedder.dim)):
            terms = ", ".join(term for term, _ in embedder.top_terms(component, 6))
            print(f"  {component}: {terms}")
        return 0

    rows = []
    grid = itertools.product(
        [16, 24, 32, 48, 64, 94], [(1, 1), (1, 2)], [0.1, 0.15, 0.2, 0.4, 1.0], [0, 1]
    )
    for dim, ngram, max_df, drop in grid:
        embedder = TfidfSvdEmbedder(
            dim=dim, ngram_range=ngram, max_df=max_df, drop_components=drop
        ).fit(texts)
        scores = evaluate(embedder, texts, topics, k=args.k)
        rows.append(
            {
                "dim": dim,
                "ngram_range": list(ngram),
                "max_df": max_df,
                "drop_components": drop,
                "vocabulary": embedder.vocabulary_size,
                **scores,
            }
        )

    key = f"precision@{args.k}"
    rows.sort(key=lambda row: (-row[key], -row["mrr"]))
    print(f"{'dim':>4} {'ngram':>7} {'max_df':>7} {'drop':>5} {'vocab':>6} {key:>12} {'mrr':>7}")
    for row in rows[:15]:
        print(
            f"{row['dim']:>4} {str(tuple(row['ngram_range'])):>7} {row['max_df']:>7} "
            f"{row['drop_components']:>5} {row['vocabulary']:>6} {row[key]:>12.4f} {row['mrr']:>7.4f}"
        )
    print(f"\n({len(rows)} configurations; random baseline {rows[0]['random_baseline']})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "text_quality_sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_dir / 'text_quality_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
