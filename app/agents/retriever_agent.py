from typing import List, Optional

from app.core.domain.models import RetrievedChunk
from app.core.interfaces.reranker import Reranker
from app.retrieval.hybrid_retriever import HybridRetriever


class RetrieverAgent:
    """Owns retrieval. Runs hybrid (dense + BM25 + RRF) search, then hands the
    candidate set to the CrossEncoder reranker (if configured) for a final
    precision pass before the Critic grades what's left."""

    def __init__(self, hybrid_retriever: HybridRetriever, reranker: Optional[Reranker] = None):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    async def retrieve(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        # Pull a wider candidate set than we'll ultimately keep, so the
        # reranker has real precision-improving work to do.
        candidate_k = max(top_k * 3, top_k + 5)
        candidates = await self.hybrid_retriever.retrieve(
            query=query, top_k=candidate_k, document_ids=document_ids
        )

        if self.reranker and candidates:
            return self.reranker.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]
