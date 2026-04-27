"""Async client for an OpenAI-compatible chat completion API.

Works with OpenAI, OpenRouter, vLLM, GigaChat-compatible gateways — anything
that speaks the OpenAI Chat Completions protocol.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from .config import get_settings

SYSTEM_PROMPT = """\
You are a retrieval-augmented assistant. Answer the user's question using ONLY
the information from the provided context. If the context does not contain the
answer, say so honestly — do not make up facts. Cite the source filenames you
relied on at the end of your answer.
"""


def build_prompt(question: str, contexts: list[tuple[str, str]]) -> list[dict]:
    """Assemble the chat messages for an OpenAI-compatible call.

    Parameters
    ----------
    question:
        The user's question.
    contexts:
        ``(source, content)`` pairs, ordered from most to least relevant.
    """
    context_block = "\n\n".join(
        f"[Source: {source}]\n{content}" for source, content in contexts
    )
    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def make_client() -> AsyncOpenAI:
    """Build an async LLM client from settings. One client per app start."""
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


async def answer(client: AsyncOpenAI, question: str, contexts: list[tuple[str, str]]) -> str:
    settings = get_settings()
    messages = build_prompt(question, contexts)
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""
