"""The user-facing layer: text in, ranked documents out.

``SearchEngine`` wires together the three things that make a search engine:

1. an **embedder** (text -> vectors)
2. an **index** (vectors -> nearest ids)
3. a **document store** (ids -> text and metadata)

Each is independently swappable, and the composition is where the real
engineering decisions live -- which index for how much data, what happens when
documents are added after the embedder was fitted, and what gets written to
disk so the service can restart without rebuilding.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseIndex
from .flat import FlatIndex
from .ivf import IVFIndex
from .ivfpq import IVFPQIndex
from .metrics import FloatArray
from .store import DocumentStore
from .text import Embedder, TfidfSvdEmbedder

__all__ = [
    "SearchEngine",
    "SearchHit",
    "choose_embedding_dim",
    "choose_index",
    "make_index",
]

# A PQ codebook has 256 entries per subspace; fewer than this many vectors per
# centroid and the codebooks are fitted on noise.
MIN_VECTORS_FOR_PQ = 50_000
MIN_VECTORS_FOR_IVF = 10_000


@dataclass
class SearchHit:
    """One result: the document, plus why it was returned."""

    id: int
    score: float
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_embedding_dim(n_documents: int) -> int:
    """Pick an LSA dimensionality for a corpus of this size.

    LSA only denoises if it actually compresses. A truncated SVD that keeps
    nearly as many dimensions as there are documents is a rotation of TF-IDF
    space, not a reduction, and topical structure never emerges -- measured on
    the demo corpus, 94 dimensions over 96 documents scored *worse* than 48.
    Roughly half the corpus size, capped, keeps the ratio sane at both ends.
    """
    return int(min(384, max(32, n_documents // 2)))


def make_index(kind: str, dim: int, **kwargs: Any) -> BaseIndex:
    """Build an index by name: ``flat``, ``ivf`` or ``ivfpq``."""
    builders: dict[str, Any] = {"flat": FlatIndex, "ivf": IVFIndex, "ivfpq": IVFPQIndex}
    if kind not in builders:
        raise ValueError(f"unknown index kind {kind!r}; choose from {sorted(builders)}")
    return builders[kind](dim=dim, **kwargs)


def choose_index(n_vectors: int, dim: int, **kwargs: Any) -> BaseIndex:
    """Pick a reasonable index for a corpus of this size.

    The thresholds are the honest answer to "which index should I use", and the
    honest answer starts with *not* the fancy one:

    * under 10k vectors, exact search is sub-millisecond and always right.
      An approximate index here adds build time, two hyperparameters and a
      recall loss in exchange for nothing.
    * from 10k to 50k, IVF pays off: the partition saves real time and nothing
      is approximated about the vectors themselves.
    * past 50k, IVF-PQ earns its compression, and there are finally enough
      vectors per subspace to fit 256-entry codebooks properly.

    Defaults for ``nlist`` follow the usual ``4 * sqrt(n)`` heuristic, clamped
    so each cell keeps enough members to be meaningful.
    """
    if n_vectors < MIN_VECTORS_FOR_IVF:
        return make_index("flat", dim=dim, **kwargs)

    nlist = kwargs.pop("nlist", None) or int(min(4 * math.sqrt(n_vectors), max(1, n_vectors // 39)))
    nlist = max(1, min(nlist, n_vectors))
    if n_vectors < MIN_VECTORS_FOR_PQ:
        return make_index("ivf", dim=dim, nlist=nlist, **kwargs)

    # m must divide dim; walk down from the preferred value until it does.
    preferred = kwargs.pop("m", None) or max(8, min(64, dim // 4))
    m = next((candidate for candidate in range(preferred, 0, -1) if dim % candidate == 0), 1)
    return make_index("ivfpq", dim=dim, nlist=nlist, m=m, **kwargs)


class SearchEngine:
    """Text search over a vector index."""

    def __init__(
        self,
        embedder: Embedder,
        index: BaseIndex,
        store: DocumentStore | None = None,
    ) -> None:
        if embedder.dim != index.dim:
            raise ValueError(
                f"embedder produces {embedder.dim}-d vectors but the index is {index.dim}-d"
            )
        self.embedder = embedder
        self.index = index
        self.store = store or DocumentStore()

    # ----------------------------------------------------------------- build

    @classmethod
    def build(
        cls,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        embedder: Embedder | None = None,
        index: str | None = None,
        embedding_dim: int | None = None,
        **index_kwargs: Any,
    ) -> SearchEngine:
        """Fit an embedder, build an index and load the documents in one call.

        ``index=None`` lets :func:`choose_index` decide from the corpus size,
        and ``embedding_dim=None`` lets :func:`choose_embedding_dim` do the
        same for the embedder.
        """
        documents = list(texts)
        if not documents:
            raise ValueError("cannot build an engine with no documents")

        if embedder is None:
            dim = embedding_dim or choose_embedding_dim(len(documents))
            fitted: Any = TfidfSvdEmbedder(dim=dim)
            fitted.fit(documents)
            embedder = fitted

        vectors = embedder.encode(documents)
        built = (
            choose_index(len(documents), embedder.dim, **index_kwargs)
            if index is None
            else make_index(index, dim=embedder.dim, **index_kwargs)
        )
        engine = cls(embedder=embedder, index=built, store=DocumentStore())
        engine._train_and_add(vectors, documents, metadatas)
        return engine

    def _train_and_add(
        self,
        vectors: FloatArray,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None,
    ) -> np.ndarray:
        if not self.index.is_trained:
            self.index.train(vectors)
        ids = self.index.add(vectors)
        self.store.add(texts, ids=ids, metadatas=metadatas)
        return ids

    # ------------------------------------------------------------------- add

    def add(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Index more documents using the embedder as already fitted.

        The embedder is deliberately *not* refitted. Refitting would change the
        vector space, which would invalidate every vector already stored and
        every centroid learned from them -- the new documents would be
        searchable and the old ones would be noise. Incremental additions
        therefore use the frozen vocabulary and projection, and new vocabulary
        is only picked up by :meth:`rebuild`. This is the same trade-off a
        production system makes when it pins an embedding model version.
        """
        documents = list(texts)
        if not documents:
            return np.zeros(0, dtype=np.int64)
        return self._train_and_add(self.embedder.encode(documents), documents, metadatas)

    def rebuild(self, index: str | None = None, **index_kwargs: Any) -> SearchEngine:
        """Refit the embedder on everything stored and build a fresh index."""
        texts = self.store.texts()
        metadatas = [doc.metadata for doc in self.store]
        return SearchEngine.build(texts, metadatas=metadatas, index=index, **index_kwargs)

    # ---------------------------------------------------------------- search

    def search(self, query: str, k: int = 10, **search_kwargs: Any) -> list[SearchHit]:
        """Return the ``k`` documents most similar to ``query``."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        vector = self.embedder.encode([query])
        ids, scores = self.index.search(vector, k=k, **search_kwargs)

        hits: list[SearchHit] = []
        for doc_id, score in zip(ids[0], scores[0], strict=True):
            if doc_id < 0 or not math.isfinite(float(score)):
                continue  # padding from an under-full result row
            document = self.store.get(int(doc_id))
            hits.append(
                SearchHit(
                    id=int(doc_id),
                    score=round(float(score), 6),
                    text=document.text if document else "",
                    metadata=document.metadata if document else {},
                )
            )
        return hits

    # ----------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        embedder_info: dict[str, Any] = {"kind": type(self.embedder).__name__}
        vocabulary = getattr(self.embedder, "vocabulary_size", None)
        if vocabulary is not None:
            embedder_info["vocabulary_size"] = vocabulary
        return {
            "documents": len(self.store),
            "embedding_dim": self.embedder.dim,
            "embedder": embedder_info,
            "index": self.index.stats(),
        }

    # ----------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> Path:
        """Write embedder, index and documents to a directory.

        Everything in it is either ``.npz``, ``.json`` or ``.jsonl``; no
        pickles, so loading an engine built elsewhere cannot run code.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.index.save(path / "index.npz")
        self.store.save(path / "documents.jsonl")
        if isinstance(self.embedder, TfidfSvdEmbedder):
            self.embedder.save(path)
            embedder_kind = "tfidf-svd"
        else:
            embedder_kind = type(self.embedder).__name__
        (path / "engine.json").write_text(
            json.dumps(
                {
                    "embedder_kind": embedder_kind,
                    "embedding_dim": self.embedder.dim,
                    "documents": len(self.store),
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> SearchEngine:
        """Load an engine saved by :meth:`save`.

        A non-persistable embedder (a transformer, say) has to be supplied by
        the caller -- the saved metadata records which kind was used so the
        mismatch is caught rather than silently producing garbage.
        """
        path = Path(directory)
        config = json.loads((path / "engine.json").read_text())
        if embedder is None:
            if config["embedder_kind"] != "tfidf-svd":
                raise ValueError(
                    f"this engine was built with {config['embedder_kind']}, which is not "
                    "serialisable; pass an equivalent embedder to load()"
                )
            embedder = TfidfSvdEmbedder.load(path)
        index = BaseIndex.load(path / "index.npz")
        store = DocumentStore.load(path / "documents.jsonl")
        return cls(embedder=embedder, index=index, store=store)
