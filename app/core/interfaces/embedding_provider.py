from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Port for turning text into vectors. Infra impl: app/embeddings/sentence_transformer_provider.py"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents (used at ingestion time)."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (used at retrieval time)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality — needed when creating the Chroma collection."""
