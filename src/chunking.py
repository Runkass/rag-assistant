"""Split raw documents into overlapping text chunks.

A thin wrapper around ``langchain_text_splitters.RecursiveCharacterTextSplitter``.
The splitter tries a hierarchy of separators (paragraphs → sentences → words →
characters), so chunks tend to break at semantically meaningful boundaries.

The ``overlap`` between chunks lets a question that lands at the boundary still
match an embedding that includes the relevant context.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class Chunk:
    source: str          # filename or URL the chunk came from
    chunk_index: int     # ordinal of the chunk within ``source``
    content: str


def split_document(source: str, text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split a single document into ordered chunks.

    Parameters
    ----------
    source:
        Identifier of the document (file name, URL, etc.). Stored as-is in the
        DB so we can later cite or re-ingest by source.
    text:
        Full text of the document.
    chunk_size:
        Target maximum chunk length in characters.
    chunk_overlap:
        How many characters of the previous chunk to repeat in the next one.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Default separators are good for both prose and code; explicit list
        # is here only to make the intent visible in code review.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    return [
        Chunk(source=source, chunk_index=i, content=piece)
        for i, piece in enumerate(pieces)
        if piece.strip()
    ]
