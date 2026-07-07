"""
Ingestion service: Document Upload -> Text Extraction -> Cleaning ->
Chunking -> Embedding Generation -> Vector Storage, exactly matching
the RAG Pipeline diagram in the spec. This is the one place that wires
extraction, cleaning, chunking, and the vector store together — routes
call this service, they never touch those modules directly.
"""

from pathlib import Path

from app.core.config.logging_config import get_logger
from app.core.domain.models import Chunk, Document, DocumentStatus
from app.core.interfaces.repositories import DocumentRepository
from app.core.interfaces.vector_store import VectorStore
from app.rag.chunking.factory import get_chunker
from app.rag.cleaning import clean_text
from app.rag.extraction.factory import extract_text
from app.retrieval.hybrid_retriever import HybridRetriever

log = get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_store: VectorStore,
        hybrid_retriever: HybridRetriever,
    ):
        self.document_repository = document_repository
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever

    async def ingest(self, file_path: Path, original_filename: str) -> Document:
        document = Document(
            filename=original_filename,
            file_type=file_path.suffix.lower().lstrip("."),
            size_bytes=file_path.stat().st_size,
            status=DocumentStatus.UPLOADED,
        )
        await self.document_repository.create(document)

        try:
            await self.document_repository.update_status(
                document.document_id, status=DocumentStatus.PROCESSING.value
            )

            pages = extract_text(file_path)
            if not pages:
                raise ValueError("No extractable text found in the document")

            chunker = get_chunker()
            all_chunks: list[Chunk] = []
            for page in pages:
                cleaned = clean_text(page.text)
                if not cleaned:
                    continue
                for text_chunk in chunker.split(cleaned):
                    all_chunks.append(
                        Chunk(
                            document_id=document.document_id,
                            filename=original_filename,
                            content=text_chunk.text,
                            page=page.page_number,
                            chunk_index=text_chunk.chunk_index,
                        )
                    )

            if not all_chunks:
                raise ValueError("Document produced zero chunks after cleaning")

            # Batch embedding generation happens inside vector_store.add_chunks
            # (it calls embed_documents once over the whole batch, respecting
            # settings.embedding_batch_size internally).
            await self.vector_store.add_chunks(all_chunks)
            await self.hybrid_retriever.refresh_bm25_index()

            await self.document_repository.update_status(
                document.document_id, status=DocumentStatus.INDEXED.value, num_chunks=len(all_chunks)
            )
            document.status = DocumentStatus.INDEXED
            document.num_chunks = len(all_chunks)
            log.info("document_ingested", document_id=document.document_id, chunks=len(all_chunks))
            return document

        except Exception as exc:
            log.error("document_ingestion_failed", document_id=document.document_id, error=str(exc))
            await self.document_repository.update_status(
                document.document_id, status=DocumentStatus.FAILED.value, error_message=str(exc)
            )
            document.status = DocumentStatus.FAILED
            document.error_message = str(exc)
            return document

    async def delete_document(self, document_id: str) -> None:
        await self.vector_store.delete_document(document_id)
        await self.document_repository.delete(document_id)
        await self.hybrid_retriever.refresh_bm25_index()
        log.info("document_deleted", document_id=document_id)
