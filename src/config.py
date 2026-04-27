"""Centralized configuration loaded from environment variables (.env).

Using pydantic-settings instead of bare ``os.getenv`` so that:
  * required values fail fast with a useful error,
  * types are validated (e.g. TOP_K_RETRIEVE must be an int),
  * the same Settings object is the single source of truth across modules.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag"
    postgres_user: str = "rag"
    postgres_password: str = "rag"

    # Models
    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "BAAI/bge-reranker-base"
    hf_home: str = "./.hf_cache"

    # Retrieval
    top_k_retrieve: int = Field(default=20, ge=1)
    top_k_final: int = Field(default=5, ge=1)

    # LLM
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = "sk-replace-me"
    llm_model: str = "gpt-4o-mini"

    # Chunking
    chunk_size: int = Field(default=600, ge=50)
    chunk_overlap: int = Field(default=80, ge=0)

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so we don't re-read .env on every call."""
    return Settings()
