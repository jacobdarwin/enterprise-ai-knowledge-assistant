"""
BM25 sparse retrieval.

Dense embeddings are great at semantic similarity but weak at exact
keyword/entity matches (e.g. a policy code like "HR-114" or an exact
product SKU). BM25 covers that gap. We keep the index in memory and
rebuild it whenever the chunk set changes — for a laptop-scale corpus
(hundreds to low thousands of chunks) this rebuild takes milliseconds,
so there's no need for a persistent BM25 store.
"""

import re
from typing import List

from rank_bm25 import BM25Okapi

from app.core.config.logging_config import get_logger
from app.core.domain.models import Chunk, RetrievedChunk

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Simple lowercase alphanumeric tokenizer — good enough for BM25,
    which cares about term overlap, not linguistic nuance."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: List[Chunk]):
        self._chunks = chunks
        self._corpus_tokens = [_tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        log.info("bm25_index_built", num_chunks=len(chunks))

    @property
    def is_empty(self) -> bool:
        return self._bm25 is None

    def search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        if self.is_empty:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[RetrievedChunk] = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue  # no term overlap at all — not a real match
            results.append(RetrievedChunk(chunk=self._chunks[idx], bm25_score=float(scores[idx])))
        return results
