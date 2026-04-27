"""Shared pytest fixtures.

The ML-heavy modules (embeddings, reranker) need torch + sentence-transformers,
which are slow to install and pull ~470 MB of model weights at first use. None
of that belongs in unit tests, so we keep the public API but let tests inject
fakes through Protocols defined in ``src.retriever``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make ``src`` importable when pytest runs from the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ensure pydantic-settings doesn't blow up on a missing .env.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.invalid")
os.environ.setdefault("LLM_MODEL", "gpt-4o-mini")
