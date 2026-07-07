"""
Local CPU embedding provider using sentence-transformers.

Why this model class: all-MiniLM-L6-v2 (384-dim) and bge-small-en-v1.5
(384-dim) both run comfortably on CPU with <200MB RAM for the model
itself, load in a couple seconds, and need zero GPU. This is what keeps
the RTX 2050 4GB card irrelevant to this component entirely — embeddings
never touch the GPU.
"""

from functools import lru_cache
from typing import List

from app.core.config.logging_config import get_logger
from app.core.config.settings import get_settings
from app.core.interfaces.embedding_provider import EmbeddingProvider

log = get_logger(__name__)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str | None = None, device: str | None = None, batch_size: int | None = None):
        # Import lazily — sentence-transformers pulls in torch, and we don't
        # want that cost paid just for importing this module (e.g. during
        # tests of unrelated components).
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size or settings.embedding_batch_size

        log.info("embedding_model_loading", model=self.model_name, device=self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        log.info("embedding_model_loaded", model=self.model_name, dimension=self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # so dot product == cosine similarity downstream
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].tolist()


@lru_cache
def get_embedding_provider() -> "SentenceTransformerEmbeddingProvider":
    """Cached singleton — loading the model is the expensive part (a few
    seconds), so we only want to pay that cost once per process."""
    return SentenceTransformerEmbeddingProvider()
