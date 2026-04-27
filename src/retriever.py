"""Two-stage retrieval over PGVector.

Stage 1 — bi-encoder: embed the query, ask Postgres for the ``top_k_retrieve``
nearest neighbours by cosine distance. The HNSW index makes this O(log N) and
fast even on millions of chunks.

Stage 2 — cross-encoder: rerank those candidates with a slow-but-accurate
model and keep only ``top_k_final``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import psycopg
from pgvector.psycopg import register_vector

from .config import get_settings


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    chunk_index: int
    content: str
    vector_distance: float   # cosine distance returned by pgvector (lower = better)
    rerank_score: float      # cross-encoder score (higher = better)


class _Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class _Reranker(Protocol):
    def rerank(self, query: str, candidates: list[str], top_k: int) -> list[tuple[int, float]]: ...


def vector_search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    top_k: int,
) -> list[tuple[str, int, str, float]]:
    """Return ``(source, chunk_index, content, distance)`` for the nearest neighbours.

    ``<=>`` is pgvector's cosine-distance operator. Combined with the HNSW
    index built with ``vector_cosine_ops`` it uses approximate nearest-neighbour
    search.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT source, chunk_index, content, embedding <=> %s::vector AS distance
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        )
        return cur.fetchall()


def retrieve(
    query: str,
    embedder: _Embedder | None = None,
    reranker: _Reranker | None = None,
    *,
    conn: psycopg.Connection | None = None,
) -> list[RetrievedChunk]:
    """End-to-end retrieval: embed → vector search → rerank.

    The ``embedder`` and ``reranker`` are injected so tests can pass cheap
    fakes instead of pulling real models.
    """
    settings = get_settings()

    if embedder is None:
        from . import embeddings as _embeddings_mod
        embedder = _embeddings_mod
    if reranker is None:
        from . import reranker as _reranker_mod
        reranker = _reranker_mod

    own_conn = conn is None
    if own_conn:
        conn = psycopg.connect(settings.database_url)
        register_vector(conn)

    try:
        query_vec = embedder.embed_query(query)
        rows = vector_search(conn, query_vec, settings.top_k_retrieve)
    finally:
        if own_conn:
            conn.close()

    if not rows:
        return []

    contents = [row[2] for row in rows]
    ranked = reranker.rerank(query, contents, settings.top_k_final)

    return [
        RetrievedChunk(
            source=rows[idx][0],
            chunk_index=rows[idx][1],
            content=rows[idx][2],
            vector_distance=float(rows[idx][3]),
            rerank_score=score,
        )
        for idx, score in ranked
    ]
