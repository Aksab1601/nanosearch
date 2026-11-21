"""Turning text into vectors, with NumPy only.

The index does not care where vectors come from, but a search engine you cannot
type a sentence into is hard to believe in. This module provides a self-
contained text embedder so the demo and the REST service work offline, with no
model download and no dependency beyond NumPy.

The method is **LSA** (latent semantic analysis): TF-IDF term weights, then a
truncated SVD to compress the sparse term space into a few hundred dense
dimensions. It is from the 1990s and a transformer will beat it on paraphrase,
but the properties that matter structurally are the same -- dense, L2-
normalised vectors whose cosine similarity tracks topical similarity -- and
unlike a transformer it is fully inspectable: every dimension is a linear
combination of words you can print.

Three pieces are worth reading as algorithms in their own right:

* **TF-IDF.** Term frequency says what a document is about; inverse document
  frequency discounts words that are about nothing ("the"). The product is the
  oldest good idea in information retrieval and still the baseline a neural
  retriever has to beat.
* **CSR sparse storage.** A document-term matrix is >99% zeros. Compressed
  Sparse Row stores only non-zeros: ``data`` and ``indices`` arrays plus an
  ``indptr`` array of row boundaries, so row *i* is the slice
  ``indptr[i]:indptr[i+1]``. Memory goes from ``n_docs * vocab`` to ``nnz``.
* **Randomized SVD.** The exact SVD of a large matrix is unaffordable and
  unnecessary when only the top ``k`` directions are wanted. Project the matrix
  onto a few random vectors to get a subspace that captures its dominant
  action, orthonormalise, and take an exact SVD of the small projection. A
  couple of power iterations sharpen the result when the spectrum decays
  slowly. Halko, Martinsson & Tropp (2011) is the reference.

For real semantic quality, :class:`SentenceTransformerEmbedder` swaps in a
transformer behind the same two-method interface.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .metrics import FloatArray, l2_normalize

__all__ = [
    "Embedder",
    "SentenceTransformerEmbedder",
    "TfidfSvdEmbedder",
    "randomized_svd",
    "tokenize",
]

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns strings into a ``(n, dim)`` float32 matrix."""

    @property
    def dim(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> FloatArray: ...


def tokenize(text: str, ngram_range: tuple[int, int] = (1, 1)) -> list[str]:
    """Lowercase word tokens plus contiguous n-grams.

    The default is unigrams only. Bigrams recover word order -- with unigrams
    alone "machine learning" and "learning machine" are the same document, and
    "new york" is two unrelated words -- but on a small corpus almost every
    bigram occurs once, which adds thousands of features that cannot support
    any cross-document similarity. On the 96-document demo corpus bigrams
    measurably *hurt* retrieval (see ``benchmarks/bench_text_quality.py``);
    turn them on when the corpus is large enough for bigrams to repeat.
    """
    words = _TOKEN.findall(text.lower())
    low, high = ngram_range
    out: list[str] = []
    for size in range(low, high + 1):
        if size == 1:
            out.extend(words)
        else:
            out.extend(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))
    return out


def randomized_svd(
    matvec: Any,
    rmatvec: Any,
    shape: tuple[int, int],
    k: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Approximate the top ``k`` right singular vectors of a large matrix.

    The matrix is supplied as two functions rather than an array:
    ``matvec(B) -> A @ B`` and ``rmatvec(B) -> A.T @ B``. That keeps this
    routine independent of how ``A`` is stored -- here it is never
    materialised, only streamed in row blocks.

    Returns ``(singular_values, components)`` with ``components`` of shape
    ``(k, n_cols)``.
    """
    n_rows, n_cols = shape
    rank = min(k + n_oversamples, n_cols, n_rows)
    rng = np.random.default_rng(seed)

    # Random probe of the column space, then power iterations to suppress the
    # tail of the spectrum (each one multiplies the gap between the directions
    # we want and the ones we do not).
    test = rng.normal(size=(n_cols, rank)).astype(np.float32)
    sample = matvec(test)
    sample, _ = np.linalg.qr(sample)
    for _ in range(n_iter):
        sample = matvec(rmatvec(sample))
        sample, _ = np.linalg.qr(sample)

    # A is now well approximated by sample @ (sample.T @ A); the second factor
    # is small, so its exact SVD is cheap.
    projected = rmatvec(sample).T  # (rank, n_cols)
    _, singular, vt = np.linalg.svd(projected, full_matrices=False)
    return singular[:k].astype(np.float32), np.ascontiguousarray(vt[:k], dtype=np.float32)


class TfidfSvdEmbedder:
    """TF-IDF + truncated SVD (LSA), implemented on NumPy.

    Parameters
    ----------
    dim:
        Output dimensionality. 128-300 is the usual range for LSA; too few
        loses topics, too many starts fitting noise.
    max_features:
        Vocabulary cap, by document frequency. Bounds both the model size and
        the SVD cost.
    min_df:
        Drop terms appearing in fewer than this many documents -- mostly
        typos and one-off tokens, which contribute noise and bloat.
    max_df:
        Drop terms appearing in more than this *fraction* of documents. This
        parameter earns its keep: without it the top singular directions are
        dominated by "the", "a", "and" and friends, because a word in every
        document describes the average document rather than distinguishing any
        of them. IDF downweights such terms but cannot remove them, and after
        row normalisation their small weights still add up to a consistent
        direction that the SVD faithfully learns. Filtering by document
        frequency is a corpus-driven stopword list -- no hand-written word list
        to maintain, and it adapts to jargon that happens to be ubiquitous in
        *this* corpus.
    drop_components:
        Discard this many leading singular directions. Even after ``max_df``
        the first direction tends to encode "how much text is in this
        document" rather than what it is about; dropping it is standard
        practice in LSA.
    """

    def __init__(
        self,
        dim: int = 192,
        max_features: int = 8192,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 1,
        max_df: float = 0.2,
        drop_components: int = 1,
        n_iter: int = 3,
        chunk_size: int = 2048,
        seed: int = 0,
    ) -> None:
        self._dim = int(dim)
        self.max_features = int(max_features)
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.min_df = int(min_df)
        self.max_df = float(max_df)
        self.drop_components = int(drop_components)
        self.n_iter = int(n_iter)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)

        self.vocabulary_: dict[str, int] = {}
        self.idf_: FloatArray = np.zeros(0, dtype=np.float32)
        self.components_: FloatArray = np.zeros((0, 0), dtype=np.float32)
        self.singular_values_: FloatArray = np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------ properties

    @property
    def dim(self) -> int:
        return self.components_.shape[0] if self.is_fitted else self._dim

    @property
    def is_fitted(self) -> bool:
        return self.components_.size > 0

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary_)

    # ------------------------------------------------------------------- fit

    def fit(self, texts: Sequence[str]) -> TfidfSvdEmbedder:
        """Learn the vocabulary, the IDF weights and the SVD projection."""
        docs = list(texts)
        if not docs:
            raise ValueError("cannot fit an embedder on zero documents")

        tokenized = [tokenize(t, self.ngram_range) for t in docs]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))

        n_docs = len(docs)
        df_ceiling = max(1, int(self.max_df * n_docs))
        kept = [
            term
            for term, df in document_frequency.most_common(None)
            if self.min_df <= df <= df_ceiling
        ][: self.max_features]
        if not kept:
            raise ValueError(
                "no terms survived the document-frequency filters; "
                f"with {n_docs} documents, terms must appear in {self.min_df} to "
                f"{df_ceiling} of them -- raise max_df or lower min_df"
            )
        self.vocabulary_ = {term: i for i, term in enumerate(sorted(kept))}
        # Smoothed IDF, as if one document contained every term: avoids a
        # divide-by-zero for unseen terms and keeps weights finite.
        self.idf_ = np.array(
            [
                math.log((1.0 + n_docs) / (1.0 + document_frequency[term])) + 1.0
                for term in self.vocabulary_
            ],
            dtype=np.float32,
        )

        indptr, indices, data = self._build_csr(tokenized)
        vocab_size = len(self.vocabulary_)
        target = min(self._dim + self.drop_components, vocab_size, max(n_docs - 1, 1))

        def matvec(block: FloatArray) -> FloatArray:
            return self._csr_dot(indptr, indices, data, n_docs, block)

        def rmatvec(block: FloatArray) -> FloatArray:
            return self._csr_t_dot(indptr, indices, data, n_docs, vocab_size, block)

        singular, components = randomized_svd(
            matvec,
            rmatvec,
            shape=(n_docs, vocab_size),
            k=target,
            n_iter=self.n_iter,
            seed=self.seed,
        )
        keep = self.drop_components if components.shape[0] > self.drop_components else 0
        self.singular_values_ = singular[keep:]
        self.components_ = np.ascontiguousarray(components[keep:])
        return self

    def fit_transform(self, texts: Sequence[str]) -> FloatArray:
        return self.fit(texts).encode(texts)

    # ---------------------------------------------------------------- encode

    def encode(self, texts: Sequence[str]) -> FloatArray:
        """Embed documents or queries into ``(n, dim)`` unit vectors.

        Queries go through exactly the same path as documents, with the IDF
        weights and projection learned at fit time -- if they did not, query
        and document vectors would live in different spaces and their cosine
        similarity would be meaningless.
        """
        if not self.is_fitted:
            raise RuntimeError("TfidfSvdEmbedder is not fitted; call fit() first")
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)

        tokenized = [tokenize(t, self.ngram_range) for t in items]
        indptr, indices, data = self._build_csr(tokenized)
        reduced = self._csr_dot(
            indptr, indices, data, len(items), np.ascontiguousarray(self.components_.T)
        )
        return l2_normalize(reduced)

    # --------------------------------------------------- sparse plumbing

    def _build_csr(
        self, tokenized: Iterable[list[str]]
    ) -> tuple[np.ndarray, np.ndarray, FloatArray]:
        """Build an L2-normalised TF-IDF matrix in CSR form.

        ``indptr[i]:indptr[i+1]`` delimits row ``i`` inside ``indices`` and
        ``data``. Rows are normalised here so that a long document cannot
        dominate on length alone -- cosine similarity assumes it.
        """
        indptr = [0]
        all_indices: list[int] = []
        all_data: list[float] = []

        for tokens in tokenized:
            counts: Counter[int] = Counter()
            for token in tokens:
                column = self.vocabulary_.get(token)
                if column is not None:
                    counts[column] += 1
            if counts:
                cols = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
                # Sublinear TF: the tenth occurrence of a word says much less
                # than the second, so count is log-scaled rather than raw.
                values = 1.0 + np.log(
                    np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
                )
                values = values * self.idf_[cols]
                norm = float(np.linalg.norm(values))
                if norm > 0:
                    values = values / norm
                order = np.argsort(cols)
                all_indices.extend(cols[order].tolist())
                all_data.extend(values[order].tolist())
            indptr.append(len(all_indices))

        return (
            np.asarray(indptr, dtype=np.int64),
            np.asarray(all_indices, dtype=np.int64),
            np.asarray(all_data, dtype=np.float32),
        )

    def _dense_block(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: FloatArray,
        start: int,
        stop: int,
        vocab_size: int,
    ) -> FloatArray:
        """Expand CSR rows ``start:stop`` into a dense block.

        Densifying a *bounded block* and handing it to BLAS beats both a
        per-row Python loop and a full dense expansion: the block is small
        enough to stay in cache and large enough to amortise the call.
        """
        block = np.zeros((stop - start, vocab_size), dtype=np.float32)
        span = slice(int(indptr[start]), int(indptr[stop]))
        if span.stop > span.start:
            rows = np.repeat(np.arange(stop - start), np.diff(indptr[start : stop + 1]))
            block[rows, indices[span]] = data[span]
        return block

    def _csr_dot(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: FloatArray,
        n_rows: int,
        other: FloatArray,
    ) -> FloatArray:
        """``A @ other`` for CSR ``A``, in row blocks."""
        vocab_size = other.shape[0]
        out = np.zeros((n_rows, other.shape[1]), dtype=np.float32)
        for start in range(0, n_rows, self.chunk_size):
            stop = min(start + self.chunk_size, n_rows)
            block = self._dense_block(indptr, indices, data, start, stop, vocab_size)
            out[start:stop] = block @ other
        return out

    def _csr_t_dot(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: FloatArray,
        n_rows: int,
        vocab_size: int,
        other: FloatArray,
    ) -> FloatArray:
        """``A.T @ other`` for CSR ``A``, accumulated over row blocks."""
        out = np.zeros((vocab_size, other.shape[1]), dtype=np.float32)
        for start in range(0, n_rows, self.chunk_size):
            stop = min(start + self.chunk_size, n_rows)
            block = self._dense_block(indptr, indices, data, start, stop, vocab_size)
            out += block.T @ other[start:stop]
        return out

    # ----------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> Path:
        """Write the fitted embedder as ``.npz`` + JSON. Nothing is pickled."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(
            path / "embedder.npz",
            idf=self.idf_,
            components=self.components_,
            singular_values=self.singular_values_,
        )
        (path / "embedder.json").write_text(
            json.dumps(
                {
                    "kind": "tfidf-svd",
                    "dim": self._dim,
                    "max_features": self.max_features,
                    "ngram_range": list(self.ngram_range),
                    "min_df": self.min_df,
                    "max_df": self.max_df,
                    "drop_components": self.drop_components,
                    "n_iter": self.n_iter,
                    "chunk_size": self.chunk_size,
                    "seed": self.seed,
                    "vocabulary": self.vocabulary_,
                },
                indent=0,
            )
        )
        return path

    @classmethod
    def load(cls, directory: str | Path) -> TfidfSvdEmbedder:
        path = Path(directory)
        config = json.loads((path / "embedder.json").read_text())
        embedder = cls(
            dim=config["dim"],
            max_features=config["max_features"],
            ngram_range=tuple(config["ngram_range"]),  # type: ignore[arg-type]
            min_df=config["min_df"],
            max_df=config.get("max_df", 1.0),
            drop_components=config.get("drop_components", 0),
            n_iter=config["n_iter"],
            chunk_size=config["chunk_size"],
            seed=config["seed"],
        )
        embedder.vocabulary_ = {k: int(v) for k, v in config["vocabulary"].items()}
        with np.load(path / "embedder.npz", allow_pickle=False) as handle:
            embedder.idf_ = handle["idf"]
            embedder.components_ = handle["components"]
            embedder.singular_values_ = handle["singular_values"]
        return embedder

    # ----------------------------------------------------------- inspection

    def top_terms(self, component: int, n: int = 8) -> list[tuple[str, float]]:
        """The words that load most strongly on one SVD dimension.

        Useful for sanity-checking that the projection learned topics rather
        than noise, and the clearest way to show that an LSA dimension is a
        readable object -- which is exactly what a transformer embedding is
        not.
        """
        if not self.is_fitted:
            raise RuntimeError("embedder is not fitted")
        terms = list(self.vocabulary_)
        weights = self.components_[component]
        order = np.argsort(-np.abs(weights))[:n]
        return [(terms[i], round(float(weights[i]), 4)) for i in order]


class SentenceTransformerEmbedder:
    """Transformer embeddings, for when retrieval quality matters more than setup.

    Kept behind the same interface as :class:`TfidfSvdEmbedder` so the index,
    the engine and the service never learn which one they are talking to. That
    separation is the useful architectural point: the hard parts of this
    project are independent of where vectors come from, which is why the
    benchmarks can run on synthetic vectors and still mean something.

    Requires ``pip install nanosearch[transformers]`` and downloads the model
    on first use.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional path
                raise ImportError(
                    "sentence-transformers is not installed; "
                    "run `pip install nanosearch[transformers]`"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._ensure_model().get_sentence_embedding_dimension())

    def fit(self, texts: Sequence[str]) -> SentenceTransformerEmbedder:  # noqa: ARG002
        """No-op: the model is pretrained. Present so the interface matches."""
        self._ensure_model()
        return self

    def encode(self, texts: Sequence[str]) -> FloatArray:
        items = [texts] if isinstance(texts, str) else list(texts)
        vectors = self._ensure_model().encode(
            items, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)
