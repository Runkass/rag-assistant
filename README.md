# rag-assistant

Production-ready заготовка RAG-сервиса (Retrieval-Augmented Generation) поверх
локальных документов. Сделано как портфолио-проект, чтобы показать те же
кирпичики, из которых собирают реальные LLM-ассистенты: векторный индекс в
Postgres, двухстадийный retrieval, LLM с жёстким анти-галлюцинационным
промптом, async FastAPI, тесты с замоканными моделями.

```
                +------------------+
                | Markdown / docs  |
                +---------+--------+
                          |
                          v
        +-----------------+-------------------+
        |  Чанкинг (RecursiveCharacterText)   |
        +-----------------+-------------------+
                          |
                          v
        +-----------------+-------------------+
        |  Embedding (e5 multilingual, 384d)  |
        +-----------------+-------------------+
                          |
                          v
       +------------------+--------------------+
       |  PostgreSQL + pgvector (HNSW индекс)  |
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
       |  LLM (OpenAI-совместимый, RAG-промпт) |
       +------------------+--------------------+
                          |
                          v
                JSON-ответ + источники
```

## Почему именно так

- **Двухстадийный retrieval (bi-encoder + cross-encoder).** Bi-encoder быстрый
  и может сканировать всю базу, но грубоват. Cross-encoder точный, но слишком
  медленный, чтобы прогонять на каждом чанке. Поэтому достаём топ-20 векторно
  и реранкуем до топ-5 — стандартная схема.
- **pgvector + HNSW.** Эксплуатация остаётся скучной: один Postgres, одно
  расширение, отдельную векторную БД нянчить не надо. HNSW даёт ANN за
  логарифмическое время, `vector_cosine_ops` совпадает с тем, как косинусная
  близость считается у эмбеддера.
- **`UNIQUE (source, chunk_index)` + `ON CONFLICT`.** Ingest идемпотентный:
  можно перезапускать после правок документа — дубликатов чанков не появится.
- **Модели грузятся лениво, через `Protocol`.** Retriever принимает любой
  объект с методами `embed_query`/`rerank` — в тестах подставляются дешёвые
  фейки, в проде — настоящий `sentence-transformers`.
- **Async FastAPI + один LLM-клиент на `lifespan`.** Никакого пересоздания
  клиента на каждый запрос, никаких утёкших сокетов.
- **`temperature=0.2` и system-промпт, запрещающий галлюцинации.** Смысл RAG
  в том, чтобы быть привязанным к источникам, а не «творчески
  переосмысливать» их.

## Стек

- Python 3.11+
- PostgreSQL 16 + [`pgvector`](https://github.com/pgvector/pgvector)
- `sentence-transformers` (`intfloat/multilingual-e5-small` для эмбеддингов,
  `BAAI/bge-reranker-base` для cross-encoder)
- `langchain-text-splitters` для чанкинга
- FastAPI + `psycopg` 3 + async-клиент `openai` 1.x
- `pytest`, `ruff`

## Быстрый старт

```bash
# 1. Postgres c pgvector
docker compose up -d
docker compose exec postgres pg_isready -U rag    # ждём "accepting connections"

# 2. Python-окружение
python -m venv .venv
.\.venv\Scripts\Activate.ps1                       # PowerShell
pip install -r requirements.txt                    # core (~80 MB)
pip install -r requirements-ml.txt                 # torch + sentence-transformers (~3 GB)

# 3. Конфиг
cp .env.example .env
# отредактировать .env: OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL

# 4. Ingest сэмплов
python -m src.ingest samples/sample_docs

# 5. Запуск API
uvicorn src.api:app --reload
# POST http://127.0.0.1:8000/ask  {"question": "Какова миссия компании?"}
```

## Конфигурация

Все настройки лежат в `.env` и грузятся через `pydantic-settings` (см.
`src/config.py`). Ключевые:

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `POSTGRES_*` | см. compose | креды БД |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | bi-encoder, 384 dim |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | cross-encoder |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 100 | размер и нахлёст чанков (в символах) |
| `TOP_K_RETRIEVE` / `TOP_K_FINAL` | 20 / 5 | глубина выдачи до и после реранка |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` | — | любой OpenAI-совместимый эндпоинт |

## Структура

```
rag-assistant/
├── docker-compose.yml         # postgres + pgvector
├── scripts/init_db.sql        # extension, таблица, HNSW-индекс
├── src/
│   ├── config.py              # pydantic-settings
│   ├── chunking.py            # обёртка над RecursiveCharacterTextSplitter
│   ├── embeddings.py          # e5 c обязательными префиксами "passage:"/"query:"
│   ├── reranker.py            # bge cross-encoder
│   ├── retriever.py           # vector_search + rerank, дружит с DI
│   ├── llm.py                 # async OpenAI-клиент + RAG-промпт
│   ├── ingest.py              # CLI: документы → чанки → эмбеддинги → upsert
│   └── api.py                 # FastAPI-приложение
├── tests/                     # pytest, все модели замоканы
└── samples/sample_docs/       # демо-корпус
```

## Тесты

```bash
pytest          # 16 тестов, ~3 c, torch не нужен
ruff check src tests
```

Юнит-тесты **не загружают** `sentence-transformers` — они подсовывают
`FakeEmbedder` / `FakeReranker` через `Protocol`-ы retriever-а и мокают
OpenAI-клиента. CI (см. `.github/workflows/ci.yml`) ставит ровно
`requirements.txt` по той же причине.

## Что бы я сделал иначе для прода

- Вынес бы загрузку моделей эмбеддера и реранкера с request-path:
  либо прогревать на старте, либо запускать как отдельный gRPC/Triton-сервис,
  чтобы контейнер с API оставался лёгким.
- Стримил бы ответ LLM (Server-Sent Events) вместо ожидания полного
  completion — воспринимаемая пользователем задержка падает кратно.
- Подобрал бы `ef_search` у HNSW и прогнал бы recall@k на реальном golden-set;
  параметры HNSW влияют сильнее, чем принято признавать.
- Добавил бы **метаданные-фильтры** (`WHERE source LIKE ...`, ACL, версия
  документа) до векторного поиска — чисто-векторный RAG ломается на любом
  корпусе, где есть конфликтующие версии одного и того же документа.
- Прикрутил бы **LangSmith** или **W&B Weave** для трейсинга каждого
  retrieval и LLM-вызова: какие чанки достали, какие были rerank-скоры,
  принял ли пользователь ответ. Без этого систему улучшать невозможно.
- Сделал бы **eval-харнес** (golden Q→A пары + LLM-as-a-judge для свободных
  ответов) и гонял бы его в CI на каждое изменение промпта или модели.
- Аутентификация, rate limiting, структурированные логи, OpenTelemetry —
  ничему из этого не место в портфолио-проекте, но всё это я бы добавил
  в первый же день реального деплоя.
