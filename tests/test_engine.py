"""The engine: embedder + index + document store, and the index-choosing logic."""

from __future__ import annotations

import numpy as np
import pytest

from nanosearch.engine import (
    SearchEngine,
    choose_embedding_dim,
    choose_index,
    make_index,
)
from nanosearch.store import DocumentStore


class TestIndexSelection:
    def test_small_corpora_get_exact_search(self) -> None:
        # Under the threshold, approximation buys nothing and costs recall.
        assert choose_index(500, dim=64).stats()["index_type"] == "flat"

    def test_medium_corpora_get_ivf(self) -> None:
        assert choose_index(20_000, dim=64).stats()["index_type"] == "ivf"

    def test_large_corpora_get_compression(self) -> None:
        index = choose_index(200_000, dim=64)
        assert index.stats()["index_type"] == "ivfpq"

    def test_pq_subspaces_always_divide_the_dimension(self) -> None:
        # An awkward dimension must still produce a legal m, not an exception.
        for dim in (30, 31, 50, 97, 100, 127, 128, 384):
            index = choose_index(200_000, dim=dim)
            assert dim % index.stats()["m"] == 0, dim

    def test_nlist_stays_within_the_corpus_size(self) -> None:
        index = choose_index(12_000, dim=32)
        assert 1 <= index.stats()["nlist"] <= 12_000

    def test_explicit_kind_overrides_the_heuristic(self) -> None:
        assert make_index("flat", dim=8).stats()["index_type"] == "flat"

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="unknown index kind"):
            make_index("hnsw", dim=8)

    def test_embedding_dim_compresses_rather_than_rotates(self) -> None:
        # Keeping ~as many dimensions as documents is a rotation, not a
        # reduction, and measurably retrieves worse.
        assert choose_embedding_dim(96) < 96
        assert choose_embedding_dim(40) >= 32  # floor for tiny corpora
        assert choose_embedding_dim(10_000_000) == 384  # and a ceiling


class TestBuildAndSearch:
    def test_finds_an_on_topic_document(self, engine: SearchEngine) -> None:
        hits = engine.search("why is my database query slow", k=5)
        assert hits
        assert any(hit.metadata.get("topic") == "databases" for hit in hits[:3])

    def test_returns_text_and_metadata_not_just_ids(self, engine: SearchEngine) -> None:
        hit = engine.search("how do neural networks learn", k=1)[0]
        assert hit.text
        assert "topic" in hit.metadata
        assert -1.0001 <= hit.score <= 1.0001

    def test_respects_k(self, engine: SearchEngine) -> None:
        assert len(engine.search("soil and compost", k=3)) == 3

    def test_results_are_ordered_by_descending_score(self, engine: SearchEngine) -> None:
        scores = [hit.score for hit in engine.search("stars and galaxies", k=8)]
        assert scores == sorted(scores, reverse=True)

    def test_rejects_an_empty_query(self, engine: SearchEngine) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            engine.search("   ", k=5)

    def test_rejects_non_positive_k(self, engine: SearchEngine) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            engine.search("anything", k=0)

    def test_rejects_building_on_nothing(self) -> None:
        with pytest.raises(ValueError, match="no documents"):
            SearchEngine.build([])

    def test_rejects_embedder_index_dimension_mismatch(
        self, corpus: tuple[list[str], list[dict]]
    ) -> None:
        from nanosearch.text import TfidfSvdEmbedder

        texts, _ = corpus
        embedder = TfidfSvdEmbedder(dim=16, seed=0).fit(texts)
        with pytest.raises(ValueError, match="embedder produces"):
            SearchEngine(embedder=embedder, index=make_index("flat", dim=99))


class TestIncrementalAdd:
    def test_added_documents_become_searchable(self, corpus: tuple[list[str], list[dict]]) -> None:
        texts, metadatas = corpus
        engine = SearchEngine.build(texts, metadatas=metadatas)
        before = len(engine.store)

        ids = engine.add(
            ["Espresso extraction depends on grind size, dose and water temperature."],
            metadatas=[{"topic": "cooking"}],
        )
        assert len(ids) == 1
        assert len(engine.store) == before + 1

        found = [hit.id for hit in engine.search("espresso grind size", k=5)]
        assert int(ids[0]) in found

    def test_adding_nothing_is_a_no_op(self, engine: SearchEngine) -> None:
        assert engine.add([]).shape == (0,)

    def test_rebuild_refits_the_embedder(self, corpus: tuple[list[str], list[dict]]) -> None:
        texts, metadatas = corpus
        engine = SearchEngine.build(texts[:50], metadatas=metadatas[:50])
        engine.add(texts[50:], metadatas=metadatas[50:])
        rebuilt = engine.rebuild()
        assert len(rebuilt.store) == len(texts)
        # Vocabulary is learned from everything now, not just the first 50.
        assert rebuilt.embedder.vocabulary_size >= engine.embedder.vocabulary_size


class TestPersistence:
    def test_round_trip_returns_identical_results(self, engine: SearchEngine, tmp_path) -> None:
        before = [(h.id, h.score) for h in engine.search("gear ratio for climbing", k=5)]
        engine.save(tmp_path / "engine")
        restored = SearchEngine.load(tmp_path / "engine")
        after = [(h.id, h.score) for h in restored.search("gear ratio for climbing", k=5)]
        assert before == after

    def test_saves_only_data_formats(self, engine: SearchEngine, tmp_path) -> None:
        path = engine.save(tmp_path / "engine")
        suffixes = {p.suffix for p in path.iterdir()}
        assert suffixes <= {".npz", ".json", ".jsonl"}

    def test_stats_describe_all_three_layers(self, engine: SearchEngine) -> None:
        stats = engine.stats()
        assert stats["documents"] > 0
        assert stats["embedding_dim"] > 0
        assert stats["index"]["index_type"]
        assert stats["embedder"]["kind"] == "TfidfSvdEmbedder"


class TestDocumentStore:
    def test_stores_and_retrieves_by_id(self) -> None:
        store = DocumentStore()
        store.add(["first", "second"], ids=np.array([7, 9]), metadatas=[{"a": 1}, {"b": 2}])
        assert len(store) == 2
        assert store.get(7).text == "first"
        assert store.get(9).metadata == {"b": 2}
        assert store.get(123) is None

    def test_rejects_mismatched_lengths(self) -> None:
        store = DocumentStore()
        with pytest.raises(ValueError, match="texts for"):
            store.add(["only one"], ids=np.array([1, 2]))
        with pytest.raises(ValueError, match="metadata entries"):
            store.add(["a", "b"], ids=np.array([1, 2]), metadatas=[{}])

    def test_jsonl_round_trip(self, tmp_path) -> None:
        store = DocumentStore()
        store.add(["alpha", "beta"], ids=np.array([1, 2]), metadatas=[{"t": "x"}, {"t": "y"}])
        store.save(tmp_path / "docs.jsonl")
        restored = DocumentStore.load(tmp_path / "docs.jsonl")
        assert len(restored) == 2
        assert restored.get(2).metadata == {"t": "y"}

    def test_loading_a_missing_file_gives_an_empty_store(self, tmp_path) -> None:
        assert len(DocumentStore.load(tmp_path / "nope.jsonl")) == 0

    def test_from_jsonl_splits_text_from_metadata(self, tmp_path) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text('{"text": "hello", "topic": "greeting"}\n')
        texts, metadatas = DocumentStore.from_jsonl(path)
        assert texts == ["hello"]
        assert metadatas == [{"topic": "greeting"}]

    def test_from_jsonl_rejects_records_without_text(self, tmp_path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"body": "no text field"}\n')
        with pytest.raises(ValueError, match="missing the 'text' field"):
            DocumentStore.from_jsonl(path)
