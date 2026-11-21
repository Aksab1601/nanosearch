"""The command line interface.

Driven through ``main(argv)`` rather than a subprocess: same code path, no
interpreter start-up per test.
"""

from __future__ import annotations

import json

import pytest

from nanosearch.cli import main


@pytest.fixture(scope="module")
def built_index(tmp_path_factory: pytest.TempPathFactory):
    directory = tmp_path_factory.mktemp("cli") / "index"
    assert main(["build", "--out", str(directory)]) == 0
    return directory


class TestBuild:
    def test_builds_the_demo_corpus_by_default(self, built_index, capsys) -> None:
        assert (built_index / "index.npz").exists()
        assert (built_index / "documents.jsonl").exists()
        assert (built_index / "embedder.json").exists()

    def test_accepts_an_explicit_corpus_and_index_kind(self, tmp_path, capsys) -> None:
        corpus = tmp_path / "tiny.jsonl"
        corpus.write_text(
            "\n".join(
                json.dumps({"text": f"document number {i} about testing", "topic": "test"})
                for i in range(40)
            )
        )
        code = main(
            ["build", str(corpus), "--out", str(tmp_path / "idx"), "--index", "flat", "--dim", "8"]
        )
        assert code == 0
        assert "flat" in capsys.readouterr().out

    def test_rejects_a_corpus_without_a_text_field(self, tmp_path) -> None:
        corpus = tmp_path / "bad.jsonl"
        corpus.write_text('{"body": "wrong field"}\n')
        with pytest.raises(ValueError, match="missing the 'text' field"):
            main(["build", str(corpus), "--out", str(tmp_path / "idx")])


class TestSearch:
    def test_prints_ranked_results(self, built_index, capsys) -> None:
        assert main(["search", str(built_index), "why is my database slow", "-k", "3"]) == 0
        out = capsys.readouterr().out
        assert "results in" in out
        assert out.count("\n") > 3

    def test_json_output_is_machine_readable(self, built_index, capsys) -> None:
        assert main(["search", str(built_index), "compost and soil", "-k", "2", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload) == 2
        assert {"id", "score", "text", "metadata"} <= set(payload[0])

    def test_rejects_an_empty_query(self, built_index) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            main(["search", str(built_index), "   "])


class TestStats:
    def test_prints_json_statistics(self, built_index, capsys) -> None:
        assert main(["stats", str(built_index)]) == 0
        stats = json.loads(capsys.readouterr().out)
        assert stats["documents"] > 0
        assert stats["index"]["index_type"]


class TestArgumentParsing:
    def test_requires_a_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            main([])

    def test_rejects_an_unknown_index_kind(self) -> None:
        with pytest.raises(SystemExit):
            main(["build", "--index", "hnsw"])
