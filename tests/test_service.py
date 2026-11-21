"""The HTTP contract.

These tests run against the real app through FastAPI's test client, including
the startup lifespan, so they cover the thing a deployment actually exposes:
status codes, validation, and the response shape clients depend on.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nanosearch.service.app import app


@pytest.fixture(scope="module")
def client():
    # The context manager runs the lifespan, which builds the demo index once.
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_healthz_reports_a_loaded_index(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["documents"] > 0
        assert body["embedding_dim"] > 0

    def test_stats_exposes_index_internals(self, client: TestClient) -> None:
        body = client.get("/stats").json()
        assert body["index"]["index_type"]
        assert body["documents"] > 0
        assert "latency" in body

    def test_home_page_renders(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "nanosearch" in response.text

    def test_openapi_schema_is_generated(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert "/search" in schema["paths"]
        assert "/documents" in schema["paths"]


class TestSearch:
    def test_returns_ranked_results(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": "how do neural networks learn", "k": 5})
        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 5
        assert body["took_ms"] >= 0
        scores = [hit["score"] for hit in body["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_results_carry_text_and_metadata(self, client: TestClient) -> None:
        body = client.post("/search", json={"query": "braising tough cuts of meat"}).json()
        first = body["results"][0]
        assert first["text"]
        assert isinstance(first["metadata"], dict)
        assert isinstance(first["id"], int)

    def test_k_defaults_when_omitted(self, client: TestClient) -> None:
        body = client.post("/search", json={"query": "ocean currents"}).json()
        assert body["k"] == 10

    def test_rejects_an_empty_query(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": ""}).status_code == 422

    def test_rejects_a_missing_query(self, client: TestClient) -> None:
        assert client.post("/search", json={"k": 5}).status_code == 422

    def test_rejects_k_below_one(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": "x", "k": 0}).status_code == 422

    def test_rejects_an_unreasonable_k(self, client: TestClient) -> None:
        # The cap is a real limit: without it one request can ask the server to
        # rank and serialise the whole corpus.
        assert client.post("/search", json={"query": "x", "k": 10_000}).status_code == 422

    def test_rejects_nprobe_below_one(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": "x", "nprobe": 0}).status_code == 422

    def test_accepts_nprobe_on_an_exact_index(self, client: TestClient) -> None:
        # The demo corpus is small enough to get a flat index, which ignores
        # nprobe rather than failing on it.
        response = client.post("/search", json={"query": "compost", "nprobe": 4})
        assert response.status_code == 200


class TestIndexing:
    def test_adds_documents_and_makes_them_searchable(self, client: TestClient) -> None:
        text = "Mulch holds moisture in the soil and keeps weed seeds from sprouting."
        before = client.get("/healthz").json()["documents"]
        response = client.post(
            "/documents", json={"documents": [{"text": text, "metadata": {"topic": "gardening"}}]}
        )
        assert response.status_code == 201
        body = response.json()
        assert body["added"] == 1
        assert body["documents_total"] == before + 1

        # Searched with its own text, so this tests indexing and retrieval
        # without depending on the frozen vocabulary covering new words.
        results = client.post("/search", json={"query": text, "k": 5}).json()
        assert body["ids"][0] in [hit["id"] for hit in results["results"]]

    def test_out_of_vocabulary_text_is_handled_not_crashed(self, client: TestClient) -> None:
        """The documented limit of a frozen embedder, asserted rather than hoped for.

        Terms the embedder never saw at fit time carry no weight, so a
        document made only of unknown words is indexed but effectively
        unreachable by those words. The service must stay up and return a
        well-formed response instead of erroring -- and the response note says
        why.
        """
        response = client.post(
            "/documents", json={"documents": [{"text": "Zorblax quibnar frellinting."}]}
        )
        assert response.status_code == 201
        assert "unchanged" in response.json()["note"]

        search = client.post("/search", json={"query": "zorblax quibnar", "k": 3})
        assert search.status_code == 200
        assert isinstance(search.json()["results"], list)

    def test_rejects_an_empty_batch(self, client: TestClient) -> None:
        assert client.post("/documents", json={"documents": []}).status_code == 422

    def test_rejects_an_empty_document(self, client: TestClient) -> None:
        assert client.post("/documents", json={"documents": [{"text": ""}]}).status_code == 422

    def test_response_explains_the_frozen_embedder(self, client: TestClient) -> None:
        body = client.post(
            "/documents", json={"documents": [{"text": "A short additional document."}]}
        ).json()
        assert "embedder" in body["note"]


class TestLatencyTracking:
    def test_stats_reports_percentiles_after_traffic(self, client: TestClient) -> None:
        for _ in range(12):
            client.post("/search", json={"query": "river delta sediment", "k": 3})
        latency = client.get("/stats").json()["latency"]
        assert latency["requests_observed"] >= 12
        assert latency["p95_ms"] >= latency["p50_ms"]
