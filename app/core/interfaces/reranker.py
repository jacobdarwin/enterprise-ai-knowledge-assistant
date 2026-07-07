from abc import ABC, abstractmethod
from typing import List

from app.core.domain.models import RetrievedChunk


class Reranker(ABC):
    """Port for cross-encoder reranking of a candidate set. Infra impl: app/reranker/ (Phase 3)."""

    @abstractmethod
    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """Score and re-sort candidates, returning the top_k."""
