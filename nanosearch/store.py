"""The document store: what a vector id actually refers to.

An index returns ids and scores. Nobody wants ids -- they want the text. The
mapping is deliberately a separate object from the index because the two have
different shapes and different lifetimes: vectors are fixed-width numeric data
that wants to be one contiguous array, documents are variable-length strings
with arbitrary metadata. Every production vector database draws the same line,
with the index in memory and the documents in a row store.

Persistence is JSON Lines: one JSON object per line. Append-friendly,
readable with ``head``, diffable, and streamable without loading the file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Document", "DocumentStore"]


@dataclass
class Document:
    """One indexed document."""

    id: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"id": self.id, "text": self.text, "metadata": self.metadata})


class DocumentStore:
    """An id-keyed collection of documents."""

    def __init__(self) -> None:
        self._documents: dict[int, Document] = {}

    def __len__(self) -> int:
        return len(self._documents)

    def __contains__(self, doc_id: int) -> bool:
        return int(doc_id) in self._documents

    def __iter__(self) -> Iterator[Document]:
        return iter(self._documents.values())

    # ------------------------------------------------------------------- add

    def add(
        self,
        texts: Sequence[str],
        ids: Sequence[int] | np.ndarray,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Store ``texts`` under ``ids``, with optional per-document metadata."""
        id_list = [int(i) for i in np.asarray(ids).reshape(-1)]
        if len(texts) != len(id_list):
            raise ValueError(f"got {len(texts)} texts for {len(id_list)} ids")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(f"got {len(metadatas)} metadata entries for {len(texts)} texts")
        for position, (text, doc_id) in enumerate(zip(texts, id_list, strict=True)):
            self._documents[doc_id] = Document(
                id=doc_id,
                text=text,
                metadata=dict(metadatas[position]) if metadatas else {},
            )

    # ------------------------------------------------------------------- get

    def get(self, doc_id: int) -> Document | None:
        return self._documents.get(int(doc_id))

    def texts(self) -> list[str]:
        return [doc.text for doc in self._documents.values()]

    # ----------------------------------------------------------- persistence

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for document in self._documents.values():
                handle.write(document.to_json() + "\n")
        return target

    @classmethod
    def load(cls, path: str | Path) -> DocumentStore:
        store = cls()
        source = Path(path)
        if not source.exists():
            return store
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                doc_id = int(record["id"])
                store._documents[doc_id] = Document(
                    id=doc_id,
                    text=record["text"],
                    metadata=record.get("metadata", {}),
                )
        return store

    @classmethod
    def from_jsonl(cls, path: str | Path, text_field: str = "text") -> tuple[list[str], list[dict]]:
        """Read a corpus file into ``(texts, metadatas)`` for indexing.

        Accepts any JSONL where each record has a text field; everything else
        in the record becomes that document's metadata.
        """
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if text_field not in record:
                    raise ValueError(f"record is missing the {text_field!r} field: {record}")
                texts.append(str(record[text_field]))
                metadatas.append({k: v for k, v in record.items() if k != text_field})
        return texts, metadatas
