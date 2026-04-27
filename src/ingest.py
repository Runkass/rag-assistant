"""CLI: read documents from a folder, chunk them, embed, upsert into PGVector.

Usage::

    python -m src.ingest ./samples/sample_docs
    python -m src.ingest ./my-data --reset            # wipe table first

Idempotent: re-running over the same files updates existing rows thanks to the
``ON CONFLICT (source, chunk_index)`` clause.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from .chunking import Chunk, split_document
from .config import get_settings
from .embeddings import embed_passages

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_documents(folder: Path) -> list[tuple[str, str]]:
    """Return a list of ``(source_name, full_text)`` for supported files."""
    documents: list[tuple[str, str]] = []
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_SUFFIXES:
            documents.append((entry.name, entry.read_text(encoding="utf-8")))
    return documents


def upsert_chunks(
    conn: psycopg.Connection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    """Insert chunks with their embeddings, updating existing rows if any."""
    rows = [
        (c.source, c.chunk_index, c.content, emb)
        for c, emb in zip(chunks, embeddings, strict=True)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (source, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (source, chunk_index) DO UPDATE
                SET content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
            """,
            rows,
        )
    conn.commit()


def reset_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY")
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Folder with .txt/.md documents to ingest")
    parser.add_argument("--reset", action="store_true", help="Wipe the chunks table first")
    args = parser.parse_args()

    settings = get_settings()
    folder = Path(args.folder)
    if not folder.is_dir():
        parser.error(f"{folder} is not a directory")

    documents = load_documents(folder)
    if not documents:
        log.warning("No supported files in %s", folder)
        return

    all_chunks: list[Chunk] = []
    for source, text in documents:
        chunks = split_document(source, text, settings.chunk_size, settings.chunk_overlap)
        all_chunks.extend(chunks)
        log.info("Chunked %s -> %d chunks", source, len(chunks))

    log.info("Embedding %d chunks via %s", len(all_chunks), settings.embedding_model)
    embeddings = embed_passages([c.content for c in all_chunks])

    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        if args.reset:
            log.info("Resetting chunks table")
            reset_table(conn)
        upsert_chunks(conn, all_chunks, embeddings)

    log.info("Ingested %d chunks from %d documents", len(all_chunks), len(documents))


if __name__ == "__main__":
    main()
