from typing import List

from app.rag.chunking.base import ChunkingStrategy, TextChunk


class OverlapChunker(ChunkingStrategy):
    """Fixed-size windows that overlap by chunk_overlap characters.
    Reduces the "cut mid-idea" problem of FixedChunker at the cost of
    storing/embedding some duplicate text."""

    def split(self, text: str) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        stride = self.chunk_size - self.chunk_overlap
        index = 0
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            segment = text[start:end].strip()
            if segment:
                chunks.append(TextChunk(text=segment, chunk_index=index, start_char=start, end_char=end))
                index += 1
            if end == text_len:
                break
            start += stride

        return chunks
