"""Command line interface: build an index, query it, inspect it, serve it.

    nanosearch build corpus.jsonl --out ./index
    nanosearch search ./index "how do neural networks learn"
    nanosearch stats ./index
    nanosearch serve ./index

A CLI matters more than it looks: it is the difference between a library you
have to write a script to try and one someone can evaluate in thirty seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .engine import SearchEngine
from .store import DocumentStore

DEMO_CORPUS = Path(__file__).resolve().parent / "data" / "corpus.jsonl"


def _build(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus) if args.corpus else DEMO_CORPUS
    texts, metadatas = DocumentStore.from_jsonl(corpus, text_field=args.text_field)
    print(f"read {len(texts)} documents from {corpus}")

    started = time.perf_counter()
    engine = SearchEngine.build(
        texts,
        metadatas=metadatas,
        index=args.index,
        embedding_dim=args.dim,
    )
    elapsed = time.perf_counter() - started
    out = engine.save(args.out)
    print(f"built in {elapsed:.2f}s -> {out}")
    print(json.dumps(engine.stats(), indent=2))
    return 0


def _search(args: argparse.Namespace) -> int:
    engine = SearchEngine.load(args.index)
    kwargs = {"nprobe": args.nprobe} if args.nprobe else {}
    started = time.perf_counter()
    hits = engine.search(args.query, k=args.k, **kwargs)
    elapsed = (time.perf_counter() - started) * 1e3

    if args.json:
        print(json.dumps([hit.as_dict() for hit in hits], indent=2))
        return 0

    print(f'\n"{args.query}"  ->  {len(hits)} results in {elapsed:.2f} ms\n')
    for rank, hit in enumerate(hits, start=1):
        topic = hit.metadata.get("topic", "")
        label = f"  [{topic}]" if topic else ""
        print(f"{rank:2d}. {hit.score:+.4f}{label}  {hit.text}")
    return 0


def _stats(args: argparse.Namespace) -> int:
    engine = SearchEngine.load(args.index)
    print(json.dumps(engine.stats(), indent=2))
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed; run `pip install nanosearch[service]`",
            file=sys.stderr,
        )
        return 1

    import os

    if args.index:
        os.environ["NANOSEARCH_INDEX"] = str(Path(args.index).resolve())
    uvicorn.run("nanosearch.service.app:app", host=args.host, port=args.port, reload=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nanosearch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build an index from a JSONL corpus")
    build.add_argument(
        "corpus", nargs="?", default=None, help="defaults to the bundled demo corpus"
    )
    build.add_argument("--out", default="./index", help="output directory")
    build.add_argument("--index", default=None, choices=[None, "flat", "ivf", "ivfpq"])
    build.add_argument(
        "--dim", type=int, default=None, help="embedding dimensionality (default: from corpus size)"
    )
    build.add_argument("--text-field", default="text")
    build.set_defaults(func=_build)

    search = sub.add_parser("search", help="query a saved index")
    search.add_argument("index", help="directory written by `build`")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--nprobe", type=int, default=None)
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=_search)

    stats = sub.add_parser("stats", help="print index statistics")
    stats.add_argument("index")
    stats.set_defaults(func=_stats)

    serve = sub.add_parser("serve", help="run the REST service")
    serve.add_argument("index", nargs="?", default=None, help="omit to serve the demo corpus")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
