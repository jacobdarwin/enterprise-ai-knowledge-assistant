"""
Hybrid retriever: orchestrates dense vector search + BM25 sparse search,
fuses them with RRF. This is the object the LangGraph "Retriever" node
(Phase 4) will call — it depends only on the VectorStore port, so it
works unchanged whether the store is Chroma, Qdrant, or Pinecone.
"""

from typing import List, Optional

from app.core.config.logging_config import get_logger
from app.core.config.settings import get_settings
from app.core.domain.models import RetrievedChunk
from app.core.interfaces.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Index
from app.retrieval.rrf import reciprocal_rank_fusion

log = get_logger(__name__)


class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self._bm25_index: Optional[BM25Index] = None

    async def refresh_bm25_index(self) -> None:
        """Rebuild the BM25 index from the vector store's current chunk set.
        Call this after ingesting or deleting documents. Cheap enough (a
        few ms for laptop-scale corpora) to call on every mutation rather
        than maintaining incremental update logic."""
        all_chunks = await self.vector_store.get_all_chunks_for_bm25()
        self._bm25_index = BM25Index(all_chunks)

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k

        dense_results = await self.vector_store.similarity_search(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
            similarity_threshold=similarity_threshold,
        )

        if not settings.hybrid_search_enabled:
            return dense_results

        if self._bm25_index is None:
            await self.refresh_bm25_index()

        bm25_results = self._bm25_index.search(query, top_k=top_k) if self._bm25_index else []

        # Metadata filtering for BM25 (document_ids) — since BM25Index
        # doesn't know about filters, apply post-hoc.
        if document_ids:
            bm25_results = [r for r in bm25_results if r.chunk.document_id in document_ids]

        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            dense_weight=settings.dense_weight,
            bm25_weight=settings.bm25_weight,
            top_k=top_k,
        )
        log.info(
            "hybrid_retrieval_complete",
            query_len=len(query),
            dense_count=len(dense_results),
            bm25_count=len(bm25_results),
            fused_count=len(fused),
        )
        return fused
