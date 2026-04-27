"""Cross-encoder reranker.

Vector search with a bi-encoder (e5) is fast but coarse: it embeds the query
and each chunk independently and compares them in vector space. A cross-encoder
takes the (query, chunk) pair together and produces a single relevance score —
much more accurate, but also much slower, so we only run it on the small
candidate set returned by the vector search.

Default model: ``BAAI/bge-reranker-base`` (multilingual, ~280 MB).
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from .config import get_settings


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.reranker_model)


def rerank(query: str, candidates: list[str], top_k: int) -> list[tuple[int, float]]:
    """Score (query, candidate) pairs and return the indices of the top-k.

    Returns
    -------
    list of ``(original_index, score)`` tuples, sorted by descending score and
    truncated to ``top_k`` items.
    """
    if not candidates:
        return []
    pairs = [(query, candidate) for candidate in candidates]
    scores = _model().predict(pairs, show_progress_bar=False)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(idx, float(score)) for idx, score in ranked[:top_k]]
