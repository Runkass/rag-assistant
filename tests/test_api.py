"""End-to-end tests for the FastAPI app with everything below the HTTP layer mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src import api
from src.api import app, get_llm_client
from src.retriever import RetrievedChunk


@pytest.fixture
def client():
    """Build a TestClient with overridden LLM dependency."""
    fake_client = AsyncMock()
    app.dependency_overrides[get_llm_client] = lambda: fake_client
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestAsk:
    def test_returns_answer_and_sources(self, client: TestClient):
        chunks = [
            RetrievedChunk("a.md", 0, "alpha", vector_distance=0.1, rerank_score=0.9),
            RetrievedChunk("b.md", 1, "bravo", vector_distance=0.2, rerank_score=0.7),
        ]
        with patch.object(api, "retrieve", return_value=chunks), \
             patch.object(api.llm, "answer", new=AsyncMock(return_value="answer text")):
            resp = client.post("/ask", json={"question": "what?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "answer text"
        assert len(body["sources"]) == 2
        assert body["sources"][0] == {"source": "a.md", "chunk_index": 0, "rerank_score": 0.9}

    def test_empty_index_returns_polite_refusal(self, client: TestClient):
        with patch.object(api, "retrieve", return_value=[]):
            resp = client.post("/ask", json={"question": "anything"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["sources"] == []
        assert "don't have" in body["answer"].lower() or "no" in body["answer"].lower()

    def test_rejects_empty_question(self, client: TestClient):
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422  # pydantic min_length=1
