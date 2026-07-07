from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.domain.models import Chunk, RetrievedChunk


class VectorStore(ABC):
    """
    Port for persistent vector storage + similarity search.
    Infra impl: app/rag/vector_store/chroma_store.py (Phase 2).
    Swapping to Qdrant/Pinecone later = implement this interface, no
    changes needed in services or the LangGraph nodes that consume it.
    """

    @abstractmethod
    async def add_chunks(self, chunks: List[Chunk]) -> None:
        """Embed (if needed) and persist chunks with their metadata."""

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Dense vector similarity search with optional metadata filtering."""

    @abstractmethod
    async def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a document."""

    @abstractmethod
    async def get_all_chunks_for_bm25(self) -> List[Chunk]:
        """Return all chunks (used to build/refresh the in-memory BM25 index)."""
