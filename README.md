# rag-assistant

Production-shaped Retrieval-Augmented Generation (RAG) service over local
documents. Built as a portfolio piece to demonstrate the same building blocks
used in real LLM-assistants: vector index in Postgres, two-stage retrieval, an
LLM with strict anti-hallucination prompting, async FastAPI, tests with mocked
models.

```
                +------------------+
                | Markdown / docs  |
                +---------+--------+
                          |
                          v
        +-----------------+-------------------+
        |  Chunking (RecursiveCharacterText)  |
        +-----------------+-------------------+
                          |
                          v
        +-----------------+-------------------+
        |  Embedding (e5 multilingual, 384d)  |
        +-----------------+-------------------+
                          |
                          v
       +------------------+--------------------+
       |  PostgreSQL + pgvector (HNSW index)   |
       +------------------+--------------------+
                          |
       /ask  ------>  vector search top-20
                          |
                          v
       +------------------+--------------------+
       |  Reranker (bge-reranker-base, top-5)  |
       +------------------+--------------------+
                          |
                          v
       +------------------+--------------------+
       |  LLM (OpenAI-compat, RAG prompt)      |
       +------------------+--------------------+
                          |
                          v
                  JSON answer + sources
```

## Why this design

- **Two-stage retrieval (bi-encoder + cross-encoder).** A bi-encoder is fast
  enough to scan the whole corpus but rough; a cross-encoder is accurate but
  too slow to run on every chunk. Recall the top-20 with vectors, then rerank
  to top-5 — standard practice.
- **pgvector with HNSW.** Keeps the operational story boring: one Postgres
  instance, one extension, no separate vector DB to babysit. HNSW gives
  approximate-NN at log-time, ``vector_cosine_ops`` matches what the embedding
  model produces.
- **`UNIQUE (source, chunk_index)` + `ON CONFLICT`.** Ingestion is idempotent,
  so re-running it after editing a document does not duplicate chunks.
- **Models loaded lazily, behind a `Protocol`.** The retriever takes any
  object with `embed_query`/`rerank` — tests inject fakes, production gets the
  real `sentence-transformers` model.
- **Async FastAPI + a single LLM client owned by `lifespan`.** No per-request
  client construction, no leaked sockets.
- **`temperature=0.2` and a system prompt that forbids hallucination.** The
  point of RAG is to be grounded; we do not want creative reinterpretation.

## Stack

- Python 3.11+
- PostgreSQL 16 + [`pgvector`](https://github.com/pgvector/pgvector)
- `sentence-transformers` (`intfloat/multilingual-e5-small` for embeddings,
  `BAAI/bge-reranker-base` for the cross-encoder)
- `langchain-text-splitters` for chunking
- FastAPI + `psycopg` 3 + `openai` 1.x async client
- `pytest`, `ruff`

## Quick start

```bash
# 1. Postgres with pgvector
docker compose up -d
docker compose exec postgres pg_isready -U rag    # wait for "accepting connections"

# 2. Python env
python -m venv .venv
.\.venv\Scripts\Activate.ps1                       # PowerShell
pip install -r requirements.txt                    # core (~80 MB)
pip install -r requirements-ml.txt                 # torch + sentence-transformers (~3 GB)

# 3. Configure
cp .env.example .env
# edit .env: OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL

# 4. Ingest sample documents
python -m src.ingest samples/sample_docs

# 5. Serve
uvicorn src.api:app --reload
# POST http://127.0.0.1:8000/ask  {"question": "What is the company's mission?"}
```

## Configuration

All settings live in `.env` and are loaded by `pydantic-settings` (see
`src/config.py`). Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_*` | see compose | DB credentials |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | bi-encoder, 384-dim |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | cross-encoder |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 100 | character-based chunking |
| `TOP_K_RETRIEVE` / `TOP_K_FINAL` | 20 / 5 | retrieval depth before/after rerank |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` | — | any OpenAI-compatible endpoint |

## Layout

```
rag-assistant/
├── docker-compose.yml         # postgres + pgvector
├── scripts/init_db.sql        # extension, table, HNSW index
├── src/
│   ├── config.py              # pydantic-settings
│   ├── chunking.py            # RecursiveCharacterTextSplitter wrapper
│   ├── embeddings.py          # e5 with required "passage:"/"query:" prefixes
│   ├── reranker.py            # bge cross-encoder
│   ├── retriever.py           # vector_search + rerank, DI-friendly
│   ├── llm.py                 # async OpenAI-compat client + RAG prompt
│   ├── ingest.py              # CLI: read docs → chunk → embed → upsert
│   └── api.py                 # FastAPI app
├── tests/                     # pytest, all models mocked
└── samples/sample_docs/       # demo corpus
```

## Tests

```bash
pytest          # 16 tests, ~3s, no torch required
ruff check src tests
```

The unit tests do **not** load `sentence-transformers` — they inject
`FakeEmbedder` / `FakeReranker` through the retriever's `Protocol`s and mock
the OpenAI client. CI (see `.github/workflows/ci.yml`) installs only
`requirements.txt` for the same reason.

## What I would do differently for production

- Move embedding and reranker model loading off the request path entirely —
  either pre-warm at startup, or run them as a separate gRPC/Triton service so
  the API container stays small.
- Stream the LLM response (Server-Sent Events) instead of waiting for the full
  completion — the user-perceived latency drops drastically.
- Replace cosine-distance + HNSW defaults with a tuned `ef_search` and run a
  recall@k benchmark on a real golden set; HNSW parameters matter more than
  most people admit.
- Add **metadata filters** (`WHERE source LIKE ...`, ACL, document version)
  before vector search — purely-vector RAG is a footgun for any corpus with
  conflicting versions.
- Hook in **LangSmith** or **W&B Weave** for tracing every retrieval and LLM
  call: which chunks were retrieved, what the rerank scores were, whether the
  user accepted the answer. Without that you cannot improve the system.
- Add an **eval harness** (golden Q→A pairs + LLM-as-a-judge for free-form
  answers) and run it in CI on every prompt or model change.
- Authentication, rate limiting, structured logging, OpenTelemetry — none of
  which belong in a portfolio project but all of which I'd add on day one of a
  real deployment.
