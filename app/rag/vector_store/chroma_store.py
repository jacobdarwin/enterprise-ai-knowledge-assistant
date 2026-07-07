"""
ChromaDB-backed vector store.

Chroma runs embedded (no server process) and persists to a local folder
via PersistentClient — this is what makes it "free vector database" in
the truest sense: no account, no API key, no network call for search,
runs entirely on the laptop's CPU/disk.

Swapping to Qdrant/Pinecone later = write a new class implementing
`VectorStore` (app/core/interfaces/vector_store.py) and change one
factory wire-up (Phase 5) — nothing in services/ or graph/ has to change.
"""

import asyncio
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config.logging_config import get_logger
from app.core.config.settings import get_settings
from app.core.domain.models import Chunk, RetrievedChunk
from app.core.interfaces.embedding_provider import EmbeddingProvider
from app.core.interfaces.vector_store import VectorStore

log = get_logger(__name__)


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.embedding_provider = embedding_provider
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # cosine space so we can turn Chroma's returned "distance" into an
        # intuitive 0..1 similarity score (1 - distance) for thresholding.
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "chroma_store_initialized",
            persist_dir=self.persist_dir,
            collection=self.collection_name,
            existing_count=self._collection.count(),
        )

    async def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        await asyncio.to_thread(self._add_chunks_sync, chunks)

    def _add_chunks_sync(self, chunks: List[Chunk]) -> None:
        texts = [c.content for c in chunks]
        embeddings = self.embedding_provider.embed_documents(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "filename": c.filename,
                "page": c.page if c.page is not None else -1,
                "chunk_index": c.chunk_index,
                "upload_time": c.upload_time.isoformat(),
                **{k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))},
            }
            for c in chunks
        ]
        self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
        log.info("chunks_added_to_vector_store", count=len(chunks))

    async def similarity_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        return await asyncio.to_thread(
            self._similarity_search_sync, query, top_k, document_ids, similarity_threshold
        )

    def _similarity_search_sync(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]],
        similarity_threshold: Optional[float],
    ) -> List[RetrievedChunk]:
        query_embedding = self.embedding_provider.embed_query(query)
        where = {"document_id": {"$in": document_ids}} if document_ids else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, content, meta, distance in zip(ids, docs, metas, distances):
            similarity = 1.0 - distance  # cosine distance -> cosine similarity
            if similarity_threshold is not None and similarity < similarity_threshold:
                continue
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", ""),
                filename=meta.get("filename", ""),
                content=content,
                page=None if meta.get("page", -1) == -1 else meta.get("page"),
                chunk_index=meta.get("chunk_index", 0),
                metadata=meta,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, dense_score=similarity))

        return retrieved

    async def delete_document(self, document_id: str) -> None:
        await asyncio.to_thread(self._collection.delete, where={"document_id": document_id})
        log.info("document_deleted_from_vector_store", document_id=document_id)

    async def get_all_chunks_for_bm25(self) -> List[Chunk]:
        return await asyncio.to_thread(self._get_all_chunks_sync)

    def _get_all_chunks_sync(self) -> List[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        chunks: List[Chunk] = []
        for chunk_id, content, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=meta.get("document_id", ""),
                    filename=meta.get("filename", ""),
                    content=content,
                    page=None if meta.get("page", -1) == -1 else meta.get("page"),
                    chunk_index=meta.get("chunk_index", 0),
                    metadata=meta,
                )
            )
        return chunks
