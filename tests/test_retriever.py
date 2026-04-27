"""Tests for the two-stage retriever using fakes for embedder, reranker and DB.

The point: retriever logic (orchestration, top-k, mapping rerank scores back
to rows) is testable without torch, sentence-transformers, or a live Postgres.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src import retriever
from src.config import get_settings
from src.retriever import retrieve


class FakeEmbedder:
    """Returns a deterministic vector — its content does not matter, the
    retriever just hands it to the DB."""

    def __init__(self):
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.0] * 384


class FakeReranker:
    """Reverses the candidate order (so we can prove the retriever
    actually uses the reranker output, not the raw vector-search order)."""

    def rerank(self, query: str, candidates: list[str], top_k: int) -> list[tuple[int, float]]:
        reversed_indices = list(range(len(candidates)))[::-1]
        return [(idx, float(len(candidates) - rank)) for rank, idx in enumerate(reversed_indices)][:top_k]


def _fake_connection(rows: list[tuple[str, int, str, float]]) -> MagicMock:
    """Build a MagicMock that imitates psycopg's context-manager cursor API."""
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


@pytest.fixture
def sample_rows() -> list[tuple[str, int, str, float]]:
    return [
        ("a.md", 0, "alpha content", 0.10),
        ("b.md", 1, "bravo content", 0.20),
        ("c.md", 2, "charlie content", 0.30),
    ]


class TestRetrieve:
    def test_calls_embedder_with_query(self, sample_rows: Any):
        embedder = FakeEmbedder()
        result = retrieve(
            "what is alpha?",
            embedder=embedder,
            reranker=FakeReranker(),
            conn=_fake_connection(sample_rows),
        )
        assert embedder.calls == ["what is alpha?"]
        assert result, "retriever returned no chunks"

    def test_reranker_output_dictates_final_order(self, sample_rows: Any, monkeypatch: pytest.MonkeyPatch):
        # Force top_k_final = 2 to verify the truncation.
        monkeypatch.setattr(get_settings(), "top_k_final", 2, raising=False)

        chunks = retrieve(
            "q",
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
            conn=_fake_connection(sample_rows),
        )
        assert [c.source for c in chunks] == ["c.md", "b.md"]
        # Rerank scores must be monotonically non-increasing.
        scores = [c.rerank_score for c in chunks]
        assert scores == sorted(scores, reverse=True)

    def test_empty_db_returns_empty_list(self):
        result = retrieve(
            "q",
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
            conn=_fake_connection([]),
        )
        assert result == []

    def test_executes_cosine_distance_query(self, sample_rows: Any):
        conn = _fake_connection(sample_rows)
        retrieve("q", embedder=FakeEmbedder(), reranker=FakeReranker(), conn=conn)

        # Verify the SQL is the cosine-distance variant (the `<=>` operator).
        executed_sql = conn.cursor.return_value.execute.call_args.args[0]
        assert "<=>" in executed_sql
        assert "ORDER BY" in executed_sql
        assert "LIMIT" in executed_sql


class TestVectorSearch:
    def test_passes_top_k_to_sql(self, sample_rows: Any):
        conn = _fake_connection(sample_rows)
        retriever.vector_search(conn, [0.1] * 384, top_k=7)
        call = conn.cursor.return_value.execute.call_args
        # Bound parameters: (embedding, embedding, top_k).
        assert call.args[1][2] == 7
