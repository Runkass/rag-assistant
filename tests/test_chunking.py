"""Tests for the chunking layer. No external deps beyond langchain-text-splitters."""

from __future__ import annotations

from src.chunking import split_document


class TestSplitDocument:
    def test_short_text_becomes_single_chunk(self):
        chunks = split_document("note.md", "Short and sweet.", chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].source == "note.md"
        assert chunks[0].chunk_index == 0
        assert chunks[0].content == "Short and sweet."

    def test_long_text_is_split_into_multiple_chunks(self):
        # 10 paragraphs, each ~80 chars, separated by blank lines.
        text = "\n\n".join(
            f"Paragraph {i}: " + ("lorem ipsum dolor sit amet " * 4)
            for i in range(10)
        )
        chunks = split_document("doc.md", text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        # Indices must be contiguous, starting from zero.
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        # Source is preserved on every chunk.
        assert {c.source for c in chunks} == {"doc.md"}
        # No chunk grossly exceeds chunk_size — the splitter is allowed to go
        # slightly over to avoid breaking words, so we leave a small margin.
        assert all(len(c.content) <= 250 for c in chunks)

    def test_empty_chunks_are_dropped(self):
        chunks = split_document("blank.md", "\n\n\n   \n", chunk_size=100, chunk_overlap=0)
        assert chunks == []

    def test_chunks_overlap_when_requested(self):
        text = "ALPHA " * 50 + "BETA " * 50 + "GAMMA " * 50
        chunks = split_document("over.md", text, chunk_size=120, chunk_overlap=40)
        assert len(chunks) >= 2
        # With overlap > 0, neighbouring chunks must share at least one word.
        for prev, curr in zip(chunks, chunks[1:], strict=False):
            common = set(prev.content.split()) & set(curr.content.split())
            assert common, f"expected overlap between chunk {prev.chunk_index} and {curr.chunk_index}"
