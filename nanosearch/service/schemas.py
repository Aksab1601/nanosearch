"""Request and response models.

Validation lives in the schema rather than in the handlers. Pydantic rejects a
malformed request before any application code runs, which means the handlers
never have to ask whether ``k`` is a sane number -- and the same declarations
generate the OpenAPI docs at ``/docs``, so the contract cannot drift away from
what the server actually accepts.

The bounds are real limits, not decoration: ``k <= 500`` and a capped batch
size stop one request from asking the server to rank the entire corpus into a
single response.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "DocumentIn",
    "HealthResponse",
    "IndexDocumentsRequest",
    "IndexDocumentsResponse",
    "SearchRequest",
    "SearchResponse",
    "SearchResultOut",
]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Natural-language query")
    k: int = Field(10, ge=1, le=500, description="How many results to return")
    nprobe: int | None = Field(
        None,
        ge=1,
        le=65536,
        description="Cells to probe; higher means better recall and slower queries. "
        "Ignored by the exact index.",
    )


class SearchResultOut(BaseModel):
    id: int
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    k: int
    nprobe: int | None = None
    took_ms: float = Field(..., description="Server-side time for embedding plus search")
    results: list[SearchResultOut]


class DocumentIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexDocumentsRequest(BaseModel):
    documents: list[DocumentIn] = Field(..., min_length=1, max_length=1000)


class IndexDocumentsResponse(BaseModel):
    added: int
    ids: list[int]
    documents_total: int
    note: str | None = None


class HealthResponse(BaseModel):
    status: str
    documents: int
    index_type: str
    embedding_dim: int
