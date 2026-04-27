"""Sentence embeddings backed by a local ``sentence-transformers`` model.

The default model is ``intfloat/multilingual-e5-small``:
  * 384-dim output (matches the ``vector(384)`` column),
  * ~470 MB download, runs on CPU,
  * supports Russian and English well.

E5 models were trained with role prefixes:
  * ``passage: ...`` for documents you index,
  * ``query: ...`` for the user question.

Mixing the prefixes hurts recall noticeably, so this module enforces them.
"""

from __future__ import annotations

import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import get_settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Lazy-load the model exactly once per process.

    Loading takes a few seconds and pulls weights from disk, so we only do it
    when something actually asks for an embedding (lets ``import src`` stay
    cheap, which matters for tests and CI).
    """
    settings = get_settings()
    os.environ.setdefault("HF_HOME", settings.hf_home)
    return SentenceTransformer(settings.embedding_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed documents to be inserted into the vector store."""
    prefixed = [f"passage: {t}" for t in texts]
    vectors = _model().encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user query."""
    vector = _model().encode(
        f"query: {text}", normalize_embeddings=True, show_progress_bar=False
    )
    return vector.tolist()
