from typing import List

from app.rag.chunking.base import ChunkingStrategy, TextChunk

# Tried in order: prefer splitting on paragraph breaks, then sentences,
# then words, then finally raw characters as a last resort. This mirrors
# LangChain's RecursiveCharacterTextSplitter behavior without adding the
# dependency for something this self-contained.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(ChunkingStrategy):
    """
    Recommended default. Recursively splits on the largest separator that
    still produces pieces <= chunk_size, so chunks tend to end on natural
    boundaries (paragraph/sentence) rather than mid-word. Adjacent chunks
    share `chunk_overlap` characters of context.
    """

    def split(self, text: str) -> List[TextChunk]:
        raw_pieces = self._recursive_split(text, _SEPARATORS)
        merged = self._merge_with_overlap(raw_pieces)

        chunks: List[TextChunk] = []
        cursor = 0
        for i, piece in enumerate(merged):
            piece = piece.strip()
            if not piece:
                continue
            start = text.find(piece[:30], cursor) if piece else cursor
            start = max(start, 0)
            end = start + len(piece)
            chunks.append(TextChunk(text=piece, chunk_index=i, start_char=start, end_char=end))
            cursor = max(cursor, end - self.chunk_overlap)
        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Base case: hard character split.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep, *rest_separators = separators
        splits = text.split(sep) if sep else list(text)

        pieces: List[str] = []
        buffer = ""
        for part in splits:
            candidate = buffer + (sep if buffer else "") + part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    pieces.append(buffer)
                if len(part) > self.chunk_size:
                    # This single part is still too big — recurse with the
                    # next, finer-grained separator.
                    pieces.extend(self._recursive_split(part, rest_separators))
                    buffer = ""
                else:
                    buffer = part
        if buffer:
            pieces.append(buffer)

        return pieces

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """Prepend the tail of the previous (already-merged) chunk to each
        piece so consecutive chunks share context, matching chunk_overlap.
        A space is inserted at the seam so words never jam together — the
        resulting chunk may run up to `chunk_overlap` characters over
        chunk_size, which is expected: overlap is added on top of the target
        size, not carved out of it."""
        if self.chunk_overlap == 0 or len(pieces) <= 1:
            return pieces

        merged: List[str] = [pieces[0]]
        for piece in pieces[1:]:
            prev_tail = merged[-1][-self.chunk_overlap :].strip()
            seam = f"{prev_tail} {piece.strip()}" if prev_tail else piece
            merged.append(seam)
        return merged
