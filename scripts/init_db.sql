-- Enable the pgvector extension. Must be the very first thing because the
-- `vector` type is referenced in the table definition below.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk. `embedding` dimensionality (384) matches
-- intfloat/multilingual-e5-small.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT       NOT NULL,                  -- file or URL the chunk came from
    chunk_index INTEGER    NOT NULL,                  -- ordinal of the chunk inside `source`
    content     TEXT       NOT NULL,                  -- raw text of the chunk
    embedding   vector(384) NOT NULL,                 -- e5-small produces 384-d vectors
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, chunk_index)
);

-- HNSW is the modern default for ANN search in pgvector: O(log N) lookup,
-- great recall, and parameters that are easy to tune.
-- `vector_cosine_ops` = cosine distance (matches how e5 embeddings are designed
-- to be compared).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Helps when we want to delete or re-ingest everything from one source.
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source);
