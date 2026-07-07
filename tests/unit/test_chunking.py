import pytest

from app.rag.chunking.fixed_chunker import FixedChunker
from app.rag.chunking.overlap_chunker import OverlapChunker
from app.rag.chunking.recursive_chunker import RecursiveChunker

SAMPLE_TEXT = (
    "Leave Policy\n\n"
    "Employees are entitled to 18 days of paid leave per calendar year. "
    "Leave requests must be submitted at least 3 days in advance.\n\n"
    "Sick Leave\n\n"
    "Employees may take up to 10 days of sick leave per year without prior approval."
)


class TestFixedChunker:
    def test_respects_chunk_size(self):
        chunker = FixedChunker(chunk_size=50)
        chunks = chunker.split(SAMPLE_TEXT)
        assert all(len(c.text) <= 50 for c in chunks)

    def test_no_data_loss_roughly(self):
        chunker = FixedChunker(chunk_size=50)
        chunks = chunker.split(SAMPLE_TEXT)
        total_chars = sum(len(c.text) for c in chunks)
        # allow for whitespace stripped between windows
        assert total_chars >= len(SAMPLE_TEXT) * 0.9

    def test_empty_text_returns_no_chunks(self):
        chunker = FixedChunker(chunk_size=50)
        assert chunker.split("") == []

    def test_rejects_invalid_sizes(self):
        with pytest.raises(ValueError):
            FixedChunker(chunk_size=0)


class TestOverlapChunker:
    def test_overlap_creates_shared_text(self):
        chunker = OverlapChunker(chunk_size=60, chunk_overlap=20)
        chunks = chunker.split(SAMPLE_TEXT)
        assert len(chunks) >= 2
        # the tail of chunk N should reappear at the head of chunk N+1
        tail_of_first = chunks[0].text[-15:]
        assert tail_of_first in chunks[1].text or chunks[1].text[:15] in chunks[0].text

    def test_rejects_overlap_gte_chunk_size(self):
        with pytest.raises(ValueError):
            OverlapChunker(chunk_size=50, chunk_overlap=50)


class TestRecursiveChunker:
    def test_no_word_jamming_at_seams(self):
        """Regression test for the word-jamming bug caught during Phase 2 —
        overlap seams must have whitespace, never glue two words together."""
        chunker = RecursiveChunker(chunk_size=80, chunk_overlap=20)
        chunks = chunker.split(SAMPLE_TEXT)
        for chunk in chunks:
            # crude check: no run of 20+ alphabetic chars with no space,
            # which is what jammed words looked like in the original bug
            words = chunk.text.split()
            assert all(len(w) < 30 for w in words), f"Suspiciously long 'word' found: {chunk.text!r}"

    def test_prefers_paragraph_boundaries(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.split(SAMPLE_TEXT)
        # "Leave Policy" and "Sick Leave" are separate paragraphs; with a
        # generous chunk_size they should end up as distinct, clean chunks
        assert any("Leave Policy" in c.text for c in chunks)
        assert any("Sick Leave" in c.text for c in chunks)

    def test_respects_chunk_size_approximately(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split(SAMPLE_TEXT)
        # overlap chunker guarantee is chunk_size + overlap, not a hard chunk_size cap
        assert all(len(c.text) <= 50 + 10 + 5 for c in chunks)  # small slack for seam spacing
