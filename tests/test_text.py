"""The text embedder: tokenising, TF-IDF, the randomized SVD, and whether the
resulting vectors mean anything.

The last point is the one that matters. Shape and dtype tests only prove the
pipeline runs; the test that earns its place is the one asserting that
documents about the same topic end up closer together than documents about
different topics, because that is the only property the index can exploit.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanosearch.text import TfidfSvdEmbedder, randomized_svd, tokenize


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self) -> None:
        assert tokenize("Hello, World! It's 2026.") == ["hello", "world", "it", "s", "2026"]

    def test_bigrams_capture_word_order(self) -> None:
        tokens = tokenize("machine learning model", ngram_range=(1, 2))
        assert "machine learning" in tokens
        assert "learning model" in tokens
        assert "machine" in tokens

    def test_empty_text_yields_no_tokens(self) -> None:
        assert tokenize("") == []
        assert tokenize("!!! ???") == []


class TestRandomizedSvd:
    def test_approximates_the_exact_decomposition(self) -> None:
        # Against a matrix with a deliberately decaying spectrum, the top
        # singular values should come back close to the exact ones.
        generator = np.random.default_rng(0)
        left = generator.normal(size=(120, 6)).astype(np.float32)
        right = generator.normal(size=(6, 40)).astype(np.float32)
        matrix = (left * np.array([9, 7, 5, 3, 2, 1], dtype=np.float32)) @ right

        exact = np.linalg.svd(matrix, compute_uv=False)[:4]
        singular, components = randomized_svd(
            lambda b: matrix @ b,
            lambda b: matrix.T @ b,
            shape=matrix.shape,
            k=4,
            n_iter=4,
            seed=0,
        )
        np.testing.assert_allclose(singular, exact, rtol=0.05)
        assert components.shape == (4, 40)

    def test_components_are_orthonormal(self) -> None:
        generator = np.random.default_rng(1)
        matrix = generator.normal(size=(60, 25)).astype(np.float32)
        _, components = randomized_svd(
            lambda b: matrix @ b, lambda b: matrix.T @ b, shape=matrix.shape, k=5, seed=0
        )
        np.testing.assert_allclose(components @ components.T, np.eye(5), atol=1e-4)


class TestFitAndEncode:
    def test_produces_unit_vectors_of_the_requested_width(
        self, corpus: tuple[list[str], list[dict]]
    ) -> None:
        texts, _ = corpus
        embedder = TfidfSvdEmbedder(dim=32, seed=0).fit(texts)
        vectors = embedder.encode(texts)
        assert vectors.shape == (len(texts), embedder.dim)
        assert vectors.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)

    def test_same_topic_documents_are_closer_than_different_topic_ones(
        self, corpus: tuple[list[str], list[dict]]
    ) -> None:
        texts, metadatas = corpus
        topics = np.array([m["topic"] for m in metadatas])
        vectors = TfidfSvdEmbedder(dim=48, seed=0).fit_transform(texts)
        similarity = vectors @ vectors.T
        np.fill_diagonal(similarity, np.nan)

        same_topic = topics[:, None] == topics[None, :]
        np.fill_diagonal(same_topic, False)
        within = float(np.nanmean(similarity[same_topic]))
        between = float(np.nanmean(similarity[~same_topic]))
        assert within > between

    def test_retrieval_quality_beats_chance_by_a_wide_margin(
        self, corpus: tuple[list[str], list[dict]]
    ) -> None:
        """Aggregate precision@3 over a labelled query set.

        Deliberately an aggregate rather than a single query. One query's
        ranking on a 96-document LSA model is noisy -- rephrasing it can flip
        the top hit -- so asserting on one would be a flaky test dressed up as
        a quality bar. Over 8 queries the metric is stable, and the threshold
        is set well below the measured value (~0.6) while still being far
        above the 0.125 a random ranker gets on 8 topics.
        """
        texts, metadatas = corpus
        topics = [m["topic"] for m in metadatas]
        labelled = [
            ("how do neural networks learn from data", "machine-learning"),
            ("why is my sql query slow", "databases"),
            ("what happens when a massive star dies", "astronomy"),
            ("getting a good crust on a steak", "cooking"),
            ("what gearing should I use for steep hills", "cycling"),
            ("which notes make up a major chord", "music"),
            ("improving heavy clay soil", "gardening"),
            ("why is one side of a mountain dry", "geography"),
        ]

        embedder = TfidfSvdEmbedder(dim=48, seed=0).fit(texts)
        document_vectors = embedder.encode(texts)
        query_vectors = embedder.encode([query for query, _ in labelled])
        similarity = query_vectors @ document_vectors.T

        precision = 0.0
        reciprocal = 0.0
        for row, (_, expected) in enumerate(labelled):
            ranking = np.argsort(-similarity[row])
            precision += sum(1 for j in ranking[:3] if topics[j] == expected) / 3
            rank = next((r for r, j in enumerate(ranking, start=1) if topics[j] == expected), None)
            reciprocal += 1.0 / rank if rank else 0.0

        assert precision / len(labelled) > 0.4
        assert reciprocal / len(labelled) > 0.5

    def test_queries_with_no_known_words_do_not_crash(
        self, corpus: tuple[list[str], list[dict]]
    ) -> None:
        # Out-of-vocabulary input must produce a (zero) vector, not an
        # exception: a search service cannot 500 on an unusual query.
        texts, _ = corpus
        embedder = TfidfSvdEmbedder(dim=16, seed=0).fit(texts)
        vectors = embedder.encode(["zzzz qqqq xyzzy"])
        assert vectors.shape == (1, embedder.dim)
        assert np.isfinite(vectors).all()

    def test_is_deterministic_across_fits(self, corpus: tuple[list[str], list[dict]]) -> None:
        texts, _ = corpus
        first = TfidfSvdEmbedder(dim=16, seed=5).fit_transform(texts)
        second = TfidfSvdEmbedder(dim=16, seed=5).fit_transform(texts)
        np.testing.assert_allclose(first, second, atol=0)

    def test_rejects_encode_before_fit(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            TfidfSvdEmbedder().encode(["anything"])

    def test_rejects_fit_on_nothing(self) -> None:
        with pytest.raises(ValueError, match="zero documents"):
            TfidfSvdEmbedder().fit([])


class TestVocabularyFilters:
    def test_max_df_removes_ubiquitous_terms(self) -> None:
        # "the" is in every document and so distinguishes none of them.
        texts = [f"the thing number {i} is here" for i in range(20)]
        embedder = TfidfSvdEmbedder(dim=4, max_df=0.5, seed=0).fit(texts)
        assert "the" not in embedder.vocabulary_
        assert "is" not in embedder.vocabulary_

    def test_min_df_removes_one_off_terms(self) -> None:
        texts = [f"common word plus unique{i}" for i in range(20)]
        embedder = TfidfSvdEmbedder(dim=4, min_df=2, max_df=1.0, seed=0).fit(texts)
        assert "unique3" not in embedder.vocabulary_

    def test_raises_when_filters_remove_everything(self) -> None:
        with pytest.raises(ValueError, match="document-frequency filters"):
            TfidfSvdEmbedder(dim=2, min_df=50).fit(["a b c", "d e f"])

    def test_top_terms_reports_readable_dimensions(
        self, corpus: tuple[list[str], list[dict]]
    ) -> None:
        texts, _ = corpus
        embedder = TfidfSvdEmbedder(dim=16, seed=0).fit(texts)
        terms = embedder.top_terms(0, n=5)
        assert len(terms) == 5
        assert all(isinstance(term, str) for term, _ in terms)


class TestPersistence:
    def test_round_trip_reproduces_vectors_exactly(
        self, corpus: tuple[list[str], list[dict]], tmp_path
    ) -> None:
        texts, _ = corpus
        embedder = TfidfSvdEmbedder(dim=24, seed=0).fit(texts)
        before = embedder.encode(texts[:10])

        embedder.save(tmp_path)
        restored = TfidfSvdEmbedder.load(tmp_path)

        np.testing.assert_allclose(restored.encode(texts[:10]), before, atol=0)
        assert restored.vocabulary_ == embedder.vocabulary_
        assert restored.dim == embedder.dim

    def test_saved_files_contain_no_pickles(
        self, corpus: tuple[list[str], list[dict]], tmp_path
    ) -> None:
        # Loading must never be able to execute code from the file.
        texts, _ = corpus
        TfidfSvdEmbedder(dim=8, seed=0).fit(texts).save(tmp_path)
        with np.load(tmp_path / "embedder.npz", allow_pickle=False) as handle:
            assert set(handle.files) == {"idf", "components", "singular_values"}
