from app.core.config.settings import ChunkStrategy, get_settings
from app.rag.chunking.base import ChunkingStrategy
from app.rag.chunking.fixed_chunker import FixedChunker
from app.rag.chunking.overlap_chunker import OverlapChunker
from app.rag.chunking.recursive_chunker import RecursiveChunker


def get_chunker(
    strategy: ChunkStrategy | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> ChunkingStrategy:
    """Build a chunker. Falls back to configured defaults from settings
    when a param isn't explicitly supplied, so call sites can override
    per-request (e.g. a Settings page slider) without a full config reload."""
    settings = get_settings()
    strategy = strategy or settings.chunk_strategy
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if strategy == ChunkStrategy.FIXED:
        return FixedChunker(chunk_size=chunk_size)
    elif strategy == ChunkStrategy.OVERLAP:
        return OverlapChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == ChunkStrategy.RECURSIVE:
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unknown chunk strategy: {strategy}")
