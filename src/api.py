"""FastAPI service exposing the RAG pipeline over HTTP.

Endpoints
---------
GET  /health      Liveness probe (DB ping included).
POST /ask         Run retrieval + LLM and return the answer with sources.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import psycopg
from fastapi import Depends, FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from . import llm
from .config import get_settings
from .retriever import RetrievedChunk, retrieve


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceCitation(BaseModel):
    source: str
    chunk_index: int
    rerank_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build long-lived clients once at startup, dispose at shutdown.

    The OpenAI client owns an HTTP connection pool — we don't want to recreate
    it on every request.
    """
    app.state.llm_client = llm.make_client()
    yield
    await app.state.llm_client.close()


app = FastAPI(
    title="rag-assistant",
    description="Retrieval-Augmented Generation over local documents.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_llm_client() -> AsyncOpenAI:
    return app.state.llm_client


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, client: AsyncOpenAI = Depends(get_llm_client)) -> AskResponse:
    chunks: list[RetrievedChunk] = retrieve(req.question)
    if not chunks:
        return AskResponse(
            answer="I don't have any indexed documents to answer from.",
            sources=[],
        )

    contexts = [(c.source, c.content) for c in chunks]
    answer_text = await llm.answer(client, req.question, contexts)

    return AskResponse(
        answer=answer_text,
        sources=[
            SourceCitation(source=c.source, chunk_index=c.chunk_index, rerank_score=c.rerank_score)
            for c in chunks
        ],
    )
