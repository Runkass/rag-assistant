"""Tests for prompt assembly and the async LLM call (with a mocked client)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm import answer, build_prompt


class TestBuildPrompt:
    def test_returns_system_and_user_messages(self):
        messages = build_prompt("Q?", [("a.md", "alpha"), ("b.md", "bravo")])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_includes_all_sources(self):
        messages = build_prompt("Q?", [("a.md", "alpha"), ("b.md", "bravo")])
        user_text = messages[1]["content"]
        assert "[Source: a.md]" in user_text
        assert "alpha" in user_text
        assert "[Source: b.md]" in user_text
        assert "bravo" in user_text
        assert "Question: Q?" in user_text

    def test_system_message_forbids_hallucination(self):
        messages = build_prompt("Q?", [])
        system_text = messages[0]["content"]
        assert "ONLY" in system_text or "only" in system_text
        assert "do not make up" in system_text.lower()


class TestAnswer:
    @pytest.mark.asyncio
    async def test_uses_completion_response(self):
        mock_message = MagicMock()
        mock_message.content = "stubbed answer"
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=mock_message)]

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_completion)

        result = await answer(client, "Q?", [("a.md", "alpha")])

        assert result == "stubbed answer"
        # Verify temperature was set conservatively for RAG.
        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["temperature"] == 0.2
        assert kwargs["messages"][1]["role"] == "user"
