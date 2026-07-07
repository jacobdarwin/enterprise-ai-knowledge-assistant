from typing import List

from app.rag.chunking.base import ChunkingStrategy, TextChunk


class FixedChunker(ChunkingStrategy):
    """Splits text into strict chunk_size windows with zero overlap.
    Fast and predictable, but can cut sentences/ideas in half — use when
    speed matters more than retrieval quality (e.g. huge CSV/JSON dumps)."""

    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        # Fixed strategy ignores overlap by definition; force it to 0.
        super().__init__(chunk_size=chunk_size, chunk_overlap=0)

    def split(self, text: str) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        index = 0
        for start in range(0, len(text), self.chunk_size):
            end = min(start + self.chunk_size, len(text))
            segment = text[start:end].strip()
            if segment:
                chunks.append(TextChunk(text=segment, chunk_index=index, start_char=start, end_char=end))
                index += 1
        return chunks
