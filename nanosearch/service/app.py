"""REST service over a nanosearch engine.

Run it::

    uvicorn nanosearch.service.app:app --reload          # bundled demo corpus
    NANOSEARCH_INDEX=./my_index uvicorn nanosearch.service.app:app

Endpoints: ``/`` (browser demo), ``/healthz``, ``/stats``, ``/search``,
``/documents``, and ``/docs`` for the generated OpenAPI page.

Three decisions here are worth more than the code around them:

* **The engine is built once at startup, not per request.** Loading an index
  costs seconds; answering a query costs milliseconds. Doing the former inside
  a handler is the most common way a search service ends up slow.
* **Latency is recorded server-side** into a bounded ring buffer, and
  ``/stats`` reports p50/p95/p99 over it. A service you cannot see the tail
  latency of is a service you cannot operate -- and the ring buffer keeps that
  visibility at fixed memory cost.
* **``/healthz`` answers from state already in memory** and never touches the
  index. A health check that does real work fails under exactly the load it
  exists to report on.
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from ..engine import SearchEngine
from ..store import DocumentStore
from .schemas import (
    HealthResponse,
    IndexDocumentsRequest,
    IndexDocumentsResponse,
    SearchRequest,
    SearchResponse,
    SearchResultOut,
)

DEMO_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Bounded so a long-running process cannot grow without limit.
_LATENCIES: deque[float] = deque(maxlen=2000)


class ServiceState:
    """Holds the engine, so handlers never build or load one."""

    engine: SearchEngine | None = None
    source: str = "none"


state = ServiceState()


def load_engine() -> tuple[SearchEngine, str]:
    """Load the index named by ``NANOSEARCH_INDEX``, or build the demo corpus."""
    index_dir = os.environ.get("NANOSEARCH_INDEX")
    if index_dir:
        path = Path(index_dir)
        if not path.exists():
            raise RuntimeError(f"NANOSEARCH_INDEX points at {path}, which does not exist")
        return SearchEngine.load(path), f"loaded:{path}"

    texts, metadatas = DocumentStore.from_jsonl(DEMO_CORPUS)
    engine = SearchEngine.build(texts, metadatas=metadatas)
    return engine, f"demo:{DEMO_CORPUS.name}({len(texts)} docs)"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001 - FastAPI's signature
    state.engine, state.source = load_engine()
    yield
    state.engine = None


app = FastAPI(
    title="nanosearch",
    version="0.1.0",
    description="Vector search built from scratch: IVF-PQ indexing with exact re-ranking.",
    lifespan=lifespan,
)


def _engine() -> SearchEngine:
    if state.engine is None:
        # 503, not 500: the service is alive but not ready, which is what a
        # load balancer needs to know to keep traffic away until it is.
        raise HTTPException(status_code=503, detail="search engine is not loaded yet")
    return state.engine


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home() -> str:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return "<h1>nanosearch</h1><p>See <a href='/docs'>/docs</a>.</p>"
    return page.read_text(encoding="utf-8")


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    engine = _engine()
    return HealthResponse(
        status="ok",
        documents=len(engine.store),
        index_type=str(engine.index.stats().get("index_type", "unknown")),
        embedding_dim=engine.embedder.dim,
    )


@app.get("/stats")
async def stats() -> dict[str, Any]:
    engine = _engine()
    observed = np.asarray(_LATENCIES, dtype=np.float64)
    latency: dict[str, Any] = {"requests_observed": int(observed.size)}
    if observed.size:
        latency.update(
            {
                "p50_ms": round(float(np.percentile(observed, 50)), 3),
                "p95_ms": round(float(np.percentile(observed, 95)), 3),
                "p99_ms": round(float(np.percentile(observed, 99)), 3),
                "max_ms": round(float(observed.max()), 3),
            }
        )
    return {"source": state.source, "latency": latency, **engine.stats()}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    engine = _engine()
    kwargs: dict[str, Any] = {}
    if request.nprobe is not None:
        kwargs["nprobe"] = request.nprobe

    started = time.perf_counter()
    try:
        hits = engine.search(request.query, k=request.k, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    took_ms = (time.perf_counter() - started) * 1e3
    _LATENCIES.append(took_ms)

    return SearchResponse(
        query=request.query,
        k=request.k,
        nprobe=request.nprobe,
        took_ms=round(took_ms, 3),
        results=[SearchResultOut(**hit.as_dict()) for hit in hits],
    )


@app.post("/documents", response_model=IndexDocumentsResponse, status_code=201)
async def add_documents(request: IndexDocumentsRequest) -> IndexDocumentsResponse:
    """Index new documents using the embedder exactly as it was fitted.

    The response says so explicitly, because the alternative -- quietly
    refitting -- would invalidate every vector already stored.
    """
    engine = _engine()
    try:
        ids = engine.add(
            [document.text for document in request.documents],
            metadatas=[document.metadata for document in request.documents],
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IndexDocumentsResponse(
        added=len(ids),
        ids=[int(i) for i in ids],
        documents_total=len(engine.store),
        note="Indexed with the existing embedder; vocabulary learned at fit time is unchanged.",
    )
