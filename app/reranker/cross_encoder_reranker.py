"""
CrossEncoder reranking.

Dense/BM25/RRF give you a fast candidate set but rank purely on
vector similarity or term overlap. A CrossEncoder jointly encodes
(query, chunk) pairs and scores relevance directly — much more
accurate, but too slow to run over the whole corpus, so it only
reranks the top ~20-50 candidates that hybrid retrieval already
narrowed down. cross-encoder/ms-marco-MiniLM-L-6-v2 is ~90MB, runs
on CPU in well under a second for a few dozen pairs.
"""

from typing import List

from app.core.config.logging_config import get_logger
from app.core.config.settings import get_settings
from app.core.domain.models import RetrievedChunk
from app.core.interfaces.reranker import Reranker

log = get_logger(__name__)


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str | None = None, device: str | None = None):
        # Lazy import — same reasoning as the embedding provider: don't pay
        # the torch import cost unless this class is actually instantiated.
        from sentence_transformers import CrossEncoder

        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.reranker_device

        log.info("reranker_model_loading", model=self.model_name, device=self.device)
        self._model = CrossEncoder(self.model_name, device=self.device)
        log.info("reranker_model_loaded", model=self.model_name)

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(query, c.chunk.content) for c in candidates]
        scores = self._model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_k]
